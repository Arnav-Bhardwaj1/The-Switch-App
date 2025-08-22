"use client";

import { useState, useEffect, useMemo } from "react";
import { useSession } from "next-auth/react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { PostDetailModal } from '@/components/posts/PostDetailModal';
import { toast } from "sonner";
import { Book, Code, Heart, MessageCircle, Share2, Palette, Film, DollarSign, BarChart2, Plane, Camera, UserPlus, Loader2, Image as ImageIcon, Video, FileText, Filter } from 'lucide-react';
import { Mars, Venus, NonBinary } from "lucide-react";
import { cn } from "@/lib/utils";
// --- TYPE DEFINITIONS ---
import { Globe } from "lucide-react";
type Gender = "male" | "female" | "trans" | "other" | "common";
type Zone = Gender | "others";
interface Author { _id: string; name: string; username: string; profileImage?: string; gender: Gender; }
interface Comment { _id: string; author: Author; text: string; createdAt: string; }
interface Post { _id: string; author: Author; text: string; photo?: string; video?: string; topics: string[]; tags: string[]; likes: string[]; comments: Comment[]; savedBy: string[]; createdAt: string; }
type Interest = { name: string; icon: React.ReactNode; tag: string; };
type SuggestedUser = { _id: string; name: string; username: string; profileImage?: string; };
interface DashboardClientProps { session: { user: { id?: string | null; name?: string | null; email?: string | null; image?: string | null; gender?: Gender; }; }; };

// --- STATIC DATA ---
const interests: Interest[] = [
  { name: "Coding", icon: <Code size={14} />, tag: 'coding' },
  { name: "Art", icon: <Palette size={14} />, tag: 'art' },
  { name: "Fitness", icon: <Heart size={14} />, tag: 'fitness' },
  { name: "Music", icon: <BarChart2 size={14} />, tag: 'music' },
  { name: "Photography", icon: <Camera size={14} />, tag: 'photography' },
  { name: "Finance", icon: <DollarSign size={14} />, tag: 'finance' },
  { name: "Movies", icon: <Film size={14} />, tag: 'movies' },
  { name: "Travel", icon: <Plane size={14} />, tag: 'travel' },
];

// --- MAIN DASHBOARD COMPONENT ---
export default function DashboardClient({ session }: DashboardClientProps) {
  const [allPosts, setAllPosts] = useState<Post[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [learningMode, setLearningMode] = useState(false);
  const [activeZone, setActiveZone] = useState<Zone>("common");
  const [selectedInterests, setSelectedInterests] = useState<string[]>([]);
  const [modalState, setModalState] = useState({ isOpen: false, startIndex: 0 });
  const [postTypeFilter, setPostTypeFilter] = useState('all');
  const [suggestedUsers, setSuggestedUsers] = useState<SuggestedUser[]>([]);
  const [isFilterOpen, setIsFilterOpen] = useState(false);

  const userGender: Gender = session.user?.gender || "common";
  const availableZones: Zone[] = userGender === "common" ? ["common"] : [userGender, "common", "others"];

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const [postsRes, suggestionsRes] = await Promise.all([
          fetch('/api/posts'),
          fetch('/api/users/suggest')
        ]);
        if (!postsRes.ok) throw new Error("Failed to fetch posts");
        if (!suggestionsRes.ok) throw new Error("Failed to fetch suggestions");

        const postsData: Post[] = await postsRes.json();
        const suggestionsData: SuggestedUser[] = await suggestionsRes.json();

        setAllPosts(postsData);
        setSuggestedUsers(suggestionsData);
      } catch (error) {
        toast.error("Could not load data for the feed.");
        console.error(error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  const filteredPosts = useMemo(() => {
    let posts = allPosts.filter(p => {
      const authorGender = p.author?.gender || "common";
      let inZone = false;
      if (activeZone === 'common') {
        inZone = true;
      } else if (activeZone === 'others') {
        inZone = (authorGender !== userGender && authorGender !== 'common');
      } else {
        inZone = (authorGender === activeZone || authorGender === 'common');
      }
      return inZone;
    });

    if (selectedInterests.length > 0) {
      posts = posts.filter(p =>
        selectedInterests.some(interest => (p.tags || []).includes(interest) || (p.topics || []).includes(interest))
      );
    }

    switch (postTypeFilter) {
      case 'photos': return posts.filter(p => !!p.photo);
      case 'videos': return posts.filter(p => !!p.video);
      case 'text': return posts.filter(p => !p.photo && !p.video);
      default: return posts;
    }
  }, [activeZone, selectedInterests, postTypeFilter, allPosts, userGender]);

  const toggleInterest = (interestTag: string) => {
    setSelectedInterests(prev =>
      prev.includes(interestTag)
        ? prev.filter(i => i !== interestTag)
        : [...prev, interestTag]
    );
  };

  const openPostModal = (post: Post) => {
    const indexInFilteredList = filteredPosts.findIndex(p => p._id === post._id);
    if (indexInFilteredList !== -1) {
      setModalState({ isOpen: true, startIndex: indexInFilteredList });
    }
  };

  const handlePostUpdate = (updatedPost: any) => {
    setAllPosts(currentPosts =>
      currentPosts.map(p => p._id === updatedPost._id ? updatedPost : p)
    );
  };

  const handleLikeOnCard = async (e: React.MouseEvent, post: Post) => {
    e.stopPropagation();
    if (!session?.user?.id) { toast.error("Please log in to like posts."); return; }

    const isLiked = post.likes.includes(session.user.id);
    const updatedPost = { ...post, likes: isLiked ? post.likes.filter(id => id !== session.user.id) : [...post.likes, session.user.id] };

    handlePostUpdate(updatedPost);

    try {
      const res = await fetch(`/api/posts/${post._id}/like`, { method: 'POST' });
      if (!res.ok) throw new Error("Server failed to process like.");
    } catch (error) {
      handlePostUpdate(post);
      toast.error("Failed to update like.");
    }
  };

  const handleFollow = async (userIdToFollow: string) => {
    const originalSuggestions = suggestedUsers;
    setSuggestedUsers(prev => prev.filter(u => u._id !== userIdToFollow));
    try {
      const res = await fetch(`/api/users/${userIdToFollow}/follow`, { method: 'POST' });
      if (!res.ok) throw new Error("Failed to follow user.");
      toast.success("User followed!");
    } catch (error) {
      toast.error("Could not follow user. Please try again.");
      setSuggestedUsers(originalSuggestions);
    }
  };
  const genderIcon =
    userGender === "male"
      ? <Mars className="h-6 w-6" />
      : userGender === "female"
      ? <Venus className="h-6 w-6" />
      : <NonBinary className="h-6 w-6" />;


  return (
    <>
      <div className="bg-gray-50 min-h-screen font-sans">
        <header className="sticky top-16 md:top-0 z-10 bg-white/80 backdrop-blur-lg border-b mb-4 sm:mb-0">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-3 h-auto sm:h-16 py-3 sm:py-0">
              {/* Mobile controls */}
              <div className="flex sm:hidden items-center justify-between w-full gap-3">
                <Select value={activeZone} onValueChange={(val) => setActiveZone(val as Zone)}>
                  <SelectTrigger className="flex-1 rounded-full">
                    <SelectValue placeholder="Select zone" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableZones.map((zone) => (
                      <SelectItem key={zone} value={zone}>
                        {zone === 'others' ? 'Other Gender' : zone}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <div className="flex items-center gap-2">
                  <Button
                    size="icon"
                    variant={learningMode ? "default" : "outline"}
                    onClick={() => setLearningMode(!learningMode)}
                    aria-label="Toggle Learning Mode"
                  >
                    <Book size={16} />
                  </Button>
                  <Button
                    size="icon"
                    variant="outline"
                    onClick={() => setIsFilterOpen(true)}
                    aria-label="Open Filters"
                  >
                    <Filter size={16} />
                  </Button>
                </div>
              </div>

              {/* Desktop controls */}
              <div className="hidden sm:flex items-center justify-between w-full">
                {/* Zone Toggle Group */}
                <ToggleGroup
                  type="single"
                  value={activeZone}
                  onValueChange={(val) => val && setActiveZone(val as Zone)}
                  className="flex flex-wrap justify-center sm:justify-start gap-3"
                >
                  {availableZones.map((zone) => (
                    <ToggleGroupItem
                      key={zone}
                      value={zone}
                      className="capitalize px-5 py-2 text-sm font-medium rounded-full border border-gray-200 shadow-sm 
                             data-[state=on]:bg-blue-500 data-[state=on]:text-white data-[state=on]:border-blue-500
                             hover:bg-blue-50 hover:text-blue-600 transition-all"
                    >
                      {zone === 'others' ? 'Other Gender' : `${zone}`} Zone
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>

                {/* Learning Mode Button */}
                <Button
                  variant={learningMode ? "default" : "outline"}
                  onClick={() => setLearningMode(!learningMode)}
                  className="gap-2 rounded-full shadow-sm transition-all hover:shadow-md"
                >
                  <Book size={16} />
                  Learning Mode
                </Button>
              </div>
            </div>
          </div>
        </header>

        <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 mt-4 sm:mt-0">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            <div className="hidden lg:col-span-1 xl:col-span-2"></div>
            <div className="lg:col-span-10 xl:col-span-6 space-y-6">
              <Tabs value={postTypeFilter} onValueChange={setPostTypeFilter}>
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="all">All</TabsTrigger>
                  <TabsTrigger value="photos" className="flex items-center gap-2"><ImageIcon className="w-4 h-4" />Photos</TabsTrigger>
                  <TabsTrigger value="videos" className="flex items-center gap-2"><Video className="w-4 h-4" />Videos</TabsTrigger>
                  <TabsTrigger value="text" className="flex items-center gap-2"><FileText className="w-4 h-4" />Text</TabsTrigger>
                </TabsList>
              </Tabs>
              {isLoading ? (
                <div className="flex justify-center py-20"><Loader2 className="w-10 h-10 animate-spin text-indigo-600" /></div>
              ) : filteredPosts.length > 0 ? (
                filteredPosts.map(post =>
                  <PostCard
                    key={post._id}
                    post={post}
                    currentUserId={session.user?.id}
                    onCardClick={() => openPostModal(post)}
                    onLikeClick={(e) => handleLikeOnCard(e, post)}
                    readOnly={activeZone === 'others'}
                  />
                )
              ) : (
                <div className="text-center py-20 bg-white rounded-lg shadow-sm">
                  <p className="text-gray-600 font-semibold">No Posts Found</p>
                  <p className="text-sm text-gray-400 mt-2">Try changing your zone or clearing some filters!</p>
                </div>
              )}
            </div>
            <aside className="lg:col-span-12 xl:col-span-4">
              <div className="sticky top-24 space-y-6">
                {/* Mobile filter opener mirrors header button for convenience */}
                <div className="sm:hidden">
                  <Button variant="outline" className="w-full justify-center gap-2" onClick={() => setIsFilterOpen(true)}>
                    <Filter size={16} /> Open Filters
                  </Button>
                </div>
                <Card className="hidden sm:block">
                  <CardHeader><CardTitle>Filter by Interests</CardTitle></CardHeader>
                  <CardContent className="flex flex-wrap gap-2">
                    {interests.map((interest) => (
                      <Button key={interest.name} variant={selectedInterests.includes(interest.tag) ? "default" : "outline"} onClick={() => toggleInterest(interest.tag)} className="rounded-full gap-2">
                        {interest.icon}{interest.name}
                      </Button>
                    ))}
                  </CardContent>
                </Card>
                <DiscoverUsersCard users={suggestedUsers} onFollow={handleFollow} />
              </div>
            </aside>
          </div>
        </main>
      </div>

      <PostDetailModal
        posts={filteredPosts}
        startIndex={modalState.startIndex}
        isOpen={modalState.isOpen}
        onClose={() => setModalState({ ...modalState, isOpen: false })}
        onPostUpdate={handlePostUpdate}
        readOnly={activeZone === 'others'}
      />

      {/* Filters Modal (mobile-focused, also usable on desktop) */}
      <Dialog open={isFilterOpen} onOpenChange={setIsFilterOpen}>
        <DialogContent className="max-w-md w-[92vw] max-h-[80vh] overflow-y-auto p-4">
          <div className="space-y-4">
            <div>
              <p className="text-sm font-semibold mb-2">Filter by interests</p>
              <div className="flex flex-wrap gap-2">
                {interests.map((interest) => (
                  <Button key={interest.name} variant={selectedInterests.includes(interest.tag) ? "default" : "outline"} onClick={() => toggleInterest(interest.tag)} className="rounded-full gap-2 text-sm px-3 py-1.5">
                    {interest.icon}{interest.name}
                  </Button>
                ))}
              </div>
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="outline" onClick={() => { setSelectedInterests([]); }}>Clear</Button>
              <Button onClick={() => setIsFilterOpen(false)}>Apply</Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

// --- SUB-COMPONENTS ---
const PostCard = ({ post, currentUserId, onCardClick, onLikeClick, readOnly = false }: { post: Post; currentUserId?: string | null; onCardClick: () => void; onLikeClick: (e: React.MouseEvent) => void; readOnly?: boolean; }) => {
  const isLiked = currentUserId ? post.likes.includes(currentUserId) : false;
  return (
    <Card className="bg-white hover:border-indigo-300 transition-all duration-200">
      <div className="p-4 cursor-pointer" onClick={onCardClick}>
        <div className="flex items-center gap-3">
          <Avatar>
            <AvatarImage src={readOnly ? undefined : post.author?.profileImage} alt={readOnly ? 'Anonymous' : post.author?.name} />
            <AvatarFallback>{readOnly ? 'A' : post.author?.name?.charAt(0)}</AvatarFallback>
          </Avatar>
          <div>
            <p className="font-semibold">
              {readOnly ? 'Anonymous' : (
                post.author?.username ? <a className="hover:underline" href={`/profile/${post.author.username}`}>{post.author?.name}</a> : (post.author?.name)
              )}
            </p>
            <p className="text-xs text-gray-500 capitalize">{post.author?.gender} Zone</p>
          </div>
        </div>
        <p className="text-gray-800 my-4">{post.text}</p>
        {post.photo && <img src={post.photo} alt="Post content" className="rounded-lg mb-3 max-h-[60vh] w-full object-contain bg-gray-100" />}
        {post.video && <video src={post.video} className="rounded-lg mb-3 max-h-[60vh] w-full object-contain bg-gray-100" playsInline controls />}
        <div className="flex flex-wrap gap-2">
          {(post.tags || []).map(tag => <Badge key={tag} variant="secondary" className="capitalize"># {tag}</Badge>)}
        </div>
      </div>
      <div className="flex items-center justify-between text-gray-500 border-t px-4 py-1">
        <Button variant="ghost" size="sm" className="flex items-center gap-1.5 hover:text-red-500" onClick={onLikeClick} disabled={readOnly}>
          <Heart size={16} className={isLiked ? "text-red-500 fill-current" : ""} /> {post.likes.length}
        </Button>
        <Button variant="ghost" size="sm" className="flex items-center gap-1.5 hover:text-blue-500" onClick={onCardClick}>
          <MessageCircle size={16} /> {post.comments.length}
        </Button>
        <Button variant="ghost" size="sm" className="flex items-center gap-1.5 hover:text-green-500" disabled={readOnly}>
          <Share2 size={16} /> Share
        </Button>
      </div>
    </Card>
  );
};

const DiscoverUsersCard = ({ users, onFollow }: { users: SuggestedUser[]; onFollow: (userId: string) => void; }) => (
  <Card>
    <CardHeader>
      <CardTitle>Discover Users</CardTitle>
    </CardHeader>
    <CardContent className="space-y-4">
      {users.length > 0 ? users.map(user => (
        <div key={user._id} className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Avatar>
              <AvatarImage src={user.profileImage} alt={user.name} />
              <AvatarFallback>{user.name.charAt(0)}</AvatarFallback>
            </Avatar>
            <div>
              <p className="font-semibold text-sm">{user.name}</p>
              <p className="text-xs text-gray-500">@{user.username}</p>
            </div>
          </div>
          <Button variant="outline" size="sm" className="gap-1.5" onClick={() => onFollow(user._id)}>
            <UserPlus size={14} /> Follow
          </Button>
        </div>
      )) : <p className="text-sm text-gray-500 text-center py-4">No suggestions right now. Try adding more interests to your profile!</p>}
    </CardContent>
  </Card>
);