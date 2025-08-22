'use client';

import { useSession } from "next-auth/react";
import { useEffect, useState, useRef, useMemo } from "react";
import { toast } from "sonner";
import { upload } from '@imagekit/next';

// UI Components
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, Edit3, Save, Camera, Image as ImageIcon, Video, FileText, UserPlus, UserCheck } from "lucide-react";
import { PostDetailModal } from '@/components/posts/PostDetailModal';
import { FollowListModal } from '@/components/modals/FollowListModal';

// --- TYPE INTERFACES ---
interface Author { _id: string; name: string; username: string; profileImage?: string; }
interface Comment { _id: string; author: Author; text: string; createdAt: string; }
interface Post { _id: string; text: string; photo?: string; video?: string; createdAt: string; author: Author; likes: string[]; savedBy: string[]; comments: Comment[]; }
interface ProfileUser { _id: string; name: string; username: string; email: string; bio: string; profileImage: string; interests: string[]; posts: Post[]; followers: any[]; following: any[]; }

// --- HELPER COMPONENT for rendering individual posts in the grid ---
const PostCard = ({ post, onClick }: { post: Post, onClick: () => void }) => {
  const postType = post.photo ? 'photo' : post.video ? 'video' : 'text';
  if (postType === 'text') {
    return (
      <Card onClick={onClick} key={post._id} className="col-span-1 sm:col-span-2 md:col-span-2 cursor-pointer hover:bg-gray-50 transition">
        <CardContent className="p-6">
          <p className="text-md whitespace-pre-wrap">{post.text}</p>
          <p className="text-xs text-muted-foreground mt-4">{new Date(post.createdAt).toLocaleDateString("en-IN", { day: 'numeric', month: 'long', year: 'numeric' })}</p>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card onClick={onClick} key={post._id} className="overflow-hidden group relative cursor-pointer">
      <div className="absolute top-2 right-2 bg-black/50 p-1.5 rounded-full z-10">
        {postType === 'photo' && <ImageIcon className="w-4 h-4 text-white" />}
        {postType === 'video' && <Video className="w-4 h-4 text-white" />}
      </div>
      <div className="aspect-square w-full overflow-hidden bg-gray-100">
        {post.photo && <img src={post.photo} alt="Post content" className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"/>}
        {post.video && <div className="w-full h-full flex items-center justify-center"><Video className="w-16 h-16 text-gray-400" /></div>}
      </div>
      {post.text && <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center p-4"><p className="text-white text-center text-sm">{post.text}</p></div>}
    </Card>
  );
};

// --- MAIN DYNAMIC PROFILE PAGE COMPONENT ---
export default function ProfilePage({ params }: { params: { username: string } }) {
  const { data: session, status: sessionStatus, update } = useSession();
  
  const [profile, setProfile] = useState<ProfileUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isOwnProfile, setIsOwnProfile] = useState(false);
  const [isFollowing, setIsFollowing] = useState(false);
  const [isFollowLoading, setIsFollowLoading] = useState(false);
  const [modalContent, setModalContent] = useState<{title: 'Followers' | 'Following', users: any[]}|null>(null);
  const [sessionUserProfile, setSessionUserProfile] = useState<ProfileUser | null>(null);

  const [activeTab, setActiveTab] = useState('all');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedPostIndex, setSelectedPostIndex] = useState(0);
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [newImageFile, setNewImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [formData, setFormData] = useState({ name: "", username: "", bio: "", interests: [] as string[] });
  const availableInterests = ["Coding", "Design", "AI", "Music", "Art", "Gaming", "Sports"];

  useEffect(() => {
    const fetchAllData = async () => {
      if (!params.username) return;
      setIsLoading(true);

      try {
        const fetchPromises = [
          fetch(`/api/users/username/${params.username}`),
        ];

        if (session?.user?.id) {
          fetchPromises.push(fetch(`/api/users/${session.user.id}`));
        }

        const [profileRes, sessionUserRes] = await Promise.all(fetchPromises);
        
        if (!profileRes.ok) { throw new Error("Profile not found"); }
        const profileData: ProfileUser = await profileRes.json();
        setProfile(profileData);

        if(sessionUserRes && sessionUserRes.ok) {
          const sessionUserData: ProfileUser = await sessionUserRes.json();
          setSessionUserProfile(sessionUserData);
        }

        const isOwn = session?.user?.id === profileData._id;
        setIsOwnProfile(isOwn);
        
        if (isOwn) {
          setFormData({ name: profileData.name, username: profileData.username, bio: profileData.bio || "", interests: profileData.interests || [] });
        } else if (session?.user?.id) {
          const isUserFollowing = profileData.followers.some(follower => follower._id === session.user.id);
          setIsFollowing(isUserFollowing);
        }

      } catch (error: any) {
        toast.error(error.message || "Could not load profile.");
        setProfile(null);
      } finally {
        setIsLoading(false);
      }
    };
    if (sessionStatus !== 'loading') { fetchAllData(); }
  }, [params.username, sessionStatus, session?.user?.id]);

  const handleFollowToggle = async (targetUserId: string, source: 'profile' | 'modal') => {
    if (!session) { toast.error("Please log in to follow users."); return; }
    if (source === 'profile') setIsFollowLoading(true);

    const originalSessionProfile = sessionUserProfile;
    const isCurrentlyFollowing = sessionUserProfile?.following.some(u => u._id === targetUserId);

    setSessionUserProfile(prev => {
        if (!prev) return null;
        const newFollowing = isCurrentlyFollowing
            ? prev.following.filter(u => u._id !== targetUserId)
            : [...prev.following, { _id: targetUserId }]; // Add a temporary object for optimistic update
        return { ...prev, following: newFollowing };
    });
    if(source === 'profile') setIsFollowing(!isFollowing);

    try {
      await fetch(`/api/users/${targetUserId}/follow`, { method: 'POST' });
    } catch (error) {
      setSessionUserProfile(originalSessionProfile);
      if(source === 'profile') setIsFollowing(!!isCurrentlyFollowing);
      toast.error("Action failed. Please try again.");
    } finally {
      if (source === 'profile') setIsFollowLoading(false);
    }
  };

  const filteredPosts = useMemo(() => {
    if (!profile) return [];
    switch (activeTab) {
      case 'photos': return profile.posts.filter(p => !!p.photo);
      case 'videos': return profile.posts.filter(p => !!p.video);
      case 'text': return profile.posts.filter(p => !p.photo && !p.video);
      default: return profile.posts;
    }
  }, [profile, activeTab]);

  const openPostModal = (index: number) => { setSelectedPostIndex(index); setIsModalOpen(true); };
  const handlePostUpdate = (updatedPost: Post) => { setProfile(p => p ? ({...p, posts: p.posts.map(post => post._id === updatedPost._id ? updatedPost : post)}) : null); };
  const handleSave = async () => { if (!profile) return; setIsSaving(true); try { let updatedProfileImageUrl = profile.profileImage; if (newImageFile) { const uploadedUrl = await handleImageUpload(); if (uploadedUrl) { updatedProfileImageUrl = uploadedUrl; } else { setIsSaving(false); return; } } const finalFormData = { ...formData, profileImage: updatedProfileImageUrl, }; const res = await fetch(`/api/users/${profile._id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(finalFormData), }); const result = await res.json(); if (!res.ok) { throw new Error(result.message || 'Failed to update profile.'); } const updatedUser: ProfileUser = result.data; setProfile(updatedUser); setNewImageFile(null); setImagePreview(null); await update({ ...session, user: { ...session?.user, name: updatedUser.name, username: updatedUser.username, image: updatedUser.profileImage, }, }); toast.success("Profile updated successfully!"); setIsEditing(false); } catch (error: any) { console.error(error); toast.error(error.message || "An error occurred while saving."); } finally { setIsSaving(false); }};
  const handleCancel = () => { if (profile) { setFormData({ name: profile.name, username: profile.username, bio: profile.bio, interests: profile.interests, }); } setNewImageFile(null); setImagePreview(null); setIsEditing(false); };
  const handleInterestToggle = (interest: string) => { setFormData((prev) => ({...prev, interests: prev.interests.includes(interest) ? prev.interests.filter((i) => i !== interest) : [...prev.interests, interest]})); };
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => { const file = e.target.files?.[0]; if (file) { setNewImageFile(file); setImagePreview(URL.createObjectURL(file)); }};
  const handleImageUpload = async () => { if (!newImageFile) return ''; try { const authRes = await fetch('/api/imagekit-auth'); if (!authRes.ok) throw new Error("Failed to authenticate with ImageKit"); const authParams = await authRes.json(); const response = await upload({ file: newImageFile, fileName: newImageFile.name, ...authParams, }); return response.url; } catch (error) { console.error("Image upload failed:", error); toast.error("Image upload failed. Please try again."); return ''; }};

  if (isLoading || sessionStatus === 'loading') {
    return <div className="flex justify-center items-center h-screen"><Loader2 className="h-8 w-8 animate-spin" /></div>;
  }
  if (!profile) {
    return <div className="text-center mt-20 text-xl font-semibold">User '{params.username}' not found.</div>;
  }
  
  const displayImage = imagePreview || profile.profileImage || "/placeholder.png";

  return (
    <div className="max-w-4xl mx-auto p-4 md:p-6 space-y-8">
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col sm:flex-row items-center gap-6">
            <div className="relative">
              <img src={displayImage} alt="Profile" className="w-24 h-24 md:w-32 md:h-32 rounded-full object-cover border-4 border-primary" />
              {isEditing && (<> <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" className="hidden"/> <Button variant="outline" size="icon" className="absolute bottom-0 right-0 rounded-full" onClick={() => fileInputRef.current?.click()}> <Camera className="h-4 w-4" /> </Button> </>)}
            </div>
            <div className="flex-1 text-center sm:text-left">
              {isEditing ? ( <div className="space-y-2"> <Input value={formData.name} onChange={(e) => setFormData({ ...formData, name: e.target.value })} className="text-xl font-bold" placeholder="Full Name" /> <Input value={formData.username} onChange={(e) => setFormData({ ...formData, username: e.target.value })} placeholder="username" /> </div> ) : ( <> <h1 className="text-2xl md:text-3xl font-bold">{profile.name}</h1> <p className="text-muted-foreground">@{profile.username}</p> </> )}
              <p className="text-muted-foreground mt-1">{profile.email}</p>
              <div className="flex justify-center sm:justify-start gap-4 mt-3">
                <div className="text-center"><span className="font-bold">{profile.posts.length}</span><p className="text-sm text-muted-foreground">Posts</p></div>
                <button onClick={() => setModalContent({title: 'Followers', users: profile.followers})} className="text-center hover:bg-muted p-1 rounded-md transition-colors">
                  <span className="font-bold">{profile.followers.length}</span>
                  <p className="text-sm text-muted-foreground">Followers</p>
                </button>
                <button onClick={() => setModalContent({title: 'Following', users: profile.following})} className="text-center hover:bg-muted p-1 rounded-md transition-colors">
                  <span className="font-bold">{profile.following.length}</span>
                  <p className="text-sm text-muted-foreground">Following</p>
                </button>
              </div>
            </div>
            <div className="flex flex-col sm:flex-row gap-2 self-start">
              {isOwnProfile ? (
                isEditing 
                  ? <> <Button onClick={handleSave} disabled={isSaving}> {isSaving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Save className="mr-2 h-4 w-4"/>} Save </Button> <Button variant="outline" onClick={handleCancel}>Cancel</Button> </>
                  : <Button variant="outline" onClick={() => setIsEditing(true)}><Edit3 className="mr-2 h-4 w-4" /> Edit Profile</Button>
              ) : (
                <Button onClick={() => handleFollowToggle(profile._id, 'profile')} variant={isFollowing ? 'secondary' : 'default'} disabled={isFollowLoading}>
                  {isFollowLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : (isFollowing ? <UserCheck className="mr-2 h-4 w-4" /> : <UserPlus className="mr-2 h-4 w-4" />)}
                  {isFollowing ? 'Following' : 'Follow'}
                </Button>
              )}
            </div>
          </div>
          <div className="mt-6 space-y-4">
            <div> <h3 className="font-semibold text-lg">Bio</h3> {isEditing ? ( <Textarea value={formData.bio} onChange={(e) => setFormData({ ...formData, bio: e.target.value })} placeholder="Tell us about yourself..." /> ) : ( <p className="text-muted-foreground">{profile.bio || "No bio added yet."}</p> )} </div>
            <div> <h3 className="font-semibold text-lg mb-2">Interests</h3> <div className="flex flex-wrap gap-3"> {isEditing ? ( availableInterests.map((interest) => ( <div key={interest} className="flex items-center gap-2 p-2 border rounded-lg"> <Switch id={interest} checked={formData.interests.includes(interest)} onCheckedChange={() => handleInterestToggle(interest)} /> <label htmlFor={interest} className="text-sm font-medium">{interest}</label> </div> )) ) : ( profile.interests.length > 0 ? ( profile.interests.map((interest) => (<Badge key={interest} variant="secondary" className="text-base">{interest}</Badge>)) ) : <p className="text-sm text-muted-foreground">No interests added.</p> )} </div> </div>
          </div>
        </CardContent>
      </Card>

      <div>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="all">All</TabsTrigger>
            <TabsTrigger value="photos" className="flex items-center gap-2"><ImageIcon className="w-4 h-4" />Photos</TabsTrigger>
            <TabsTrigger value="videos" className="flex items-center gap-2"><Video className="w-4 h-4" />Videos</TabsTrigger>
            <TabsTrigger value="text" className="flex items-center gap-2"><FileText className="w-4 h-4" />Text</TabsTrigger>
          </TabsList>
          <div className="mt-4">
            {filteredPosts.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                {filteredPosts.map((post, index) => (
                  <PostCard key={post._id} post={post} onClick={() => openPostModal(index)} />
                ))}
              </div>
            ) : (
              <div className="text-center py-16 border-2 border-dashed rounded-lg mt-4">
                <p className="text-muted-foreground">This user hasn't posted in this category yet.</p>
              </div>
            )}
          </div>
        </Tabs>
      </div>
      
      {profile && (
         <PostDetailModal
            posts={filteredPosts}
            startIndex={selectedPostIndex}
            isOpen={isModalOpen}
            onClose={() => setIsModalOpen(false)}
            onPostUpdate={handlePostUpdate}
          />
      )}
      {modalContent && (
        <FollowListModal
          isOpen={!!modalContent}
          onClose={() => setModalContent(null)}
          title={modalContent.title}
          users={modalContent.users}
          currentUserId={session?.user?.id}
          sessionUserFollowing={sessionUserProfile?.following.map(u => u._id) || []}
          onFollowToggle={(userId) => handleFollowToggle(userId, 'modal')}
        />
      )}
    </div>
  );
}