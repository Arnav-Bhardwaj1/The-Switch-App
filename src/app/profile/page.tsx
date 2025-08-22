// 'use client'

// import { useSession } from "next-auth/react";
// import { useEffect, useState, useRef } from "react";
// import { toast } from "sonner";
// import { upload } from '@imagekit/next';

// import { Card, CardContent } from "@/components/ui/card";
// import { Button } from "@/components/ui/button";
// import { Badge } from "@/components/ui/badge";
// import { Switch } from "@/components/ui/switch";
// import { Input } from "@/components/ui/input";
// import { Textarea } from "@/components/ui/textarea";
// import { Loader2, Edit3, Save, Camera } from "lucide-react";

// // Updated User type
// interface ProfileUser {
//   _id: string;
//   name: string;
//   username: string;
//   email: string;
//   bio: string;
//   profileImage: string;
//   interests: string[];
//   followers: string[];
//   following: string[];
// }

// export default function ProfilePage() {
//   const { data: session, status, update } = useSession();
//   const [profile, setProfile] = useState<ProfileUser | null>(null);
//   const [isLoading, setIsLoading] = useState(true);
//   const [isEditing, setIsEditing] = useState(false);
//   const [isSaving, setIsSaving] = useState(false);
//   const fileInputRef = useRef<HTMLInputElement>(null);

//   // State for the new profile picture file and its preview URL
//   const [newImageFile, setNewImageFile] = useState<File | null>(null);
//   const [imagePreview, setImagePreview] = useState<string | null>(null);

//   // Form state for editing
//   const [formData, setFormData] = useState({
//     name: "",
//     username: "",
//     bio: "",
//     interests: [] as string[],
//   });

//   const availableInterests = ["Coding", "Design", "AI", "Music", "Art", "Gaming", "Sports"];

//   // Fetch the user profile from your backend
//   useEffect(() => {
//     const fetchProfile = async () => {
//       if (status === "authenticated" && session?.user?.id) {
//         setIsLoading(true);
//         try {
//           const res = await fetch(`/api/users/${session.user.id}`);
//           if (!res.ok) throw new Error("Failed to fetch profile");
//           const data: ProfileUser = await res.json();
//           setProfile(data);
//           setFormData({
//             name: data.name,
//             username: data.username,
//             bio: data.bio || "",
//             interests: data.interests || [],
//           });
//         } catch (error) {
//           console.error(error);
//           toast.error("Could not load your profile.");
//         } finally {
//           setIsLoading(false);
//         }
//       }
//     };

//     fetchProfile();
//   }, [status, session?.user?.id]);

//   // ✅ CORRECTED: The missing function is added back here.
//   const handleInterestToggle = (interest: string) => {
//     setFormData((prev) => {
//       const newInterests = prev.interests.includes(interest)
//         ? prev.interests.filter((i) => i !== interest)
//         : [...prev.interests, interest];
//       return { ...prev, interests: newInterests };
//     });
//   };

//   // Handle file selection and create a local preview
//   const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
//     const file = e.target.files?.[0];
//     if (file) {
//       setNewImageFile(file);
//       setImagePreview(URL.createObjectURL(file));
//     }
//   };

//   // Logic for uploading the image to ImageKit
//   const handleImageUpload = async () => {
//     if (!newImageFile) return '';
//     try {
//       const authRes = await fetch('/api/imagekit-auth');
//       if (!authRes.ok) throw new Error("Failed to authenticate with ImageKit");
//       const authParams = await authRes.json();

//       const response = await upload({
//         file: newImageFile,
//         fileName: newImageFile.name,
//         ...authParams,
//       });
//       return response.url;
//     } catch (error) {
//         console.error("Image upload failed:", error);
//         toast.error("Image upload failed. Please try again.");
//         return '';
//     }
//   };


//   const handleSave = async () => {
//     if (!profile) return;
//     setIsSaving(true);

//     try {
//       let updatedProfileImageUrl = profile.profileImage;
//       // If a new image was selected, upload it first
//       if (newImageFile) {
//         const uploadedUrl = await handleImageUpload();
//         if (uploadedUrl) {
//           updatedProfileImageUrl = uploadedUrl;
//         } else {
//           // Stop the save process if upload fails
//           setIsSaving(false);
//           return;
//         }
//       }

//       // Prepare final data for the PUT request
//       const finalFormData = {
//         ...formData,
//         profileImage: updatedProfileImageUrl,
//       };

//       const res = await fetch(`/api/users/${profile._id}`, {
//         method: 'PUT',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(finalFormData),
//       });

//       const result = await res.json();

//       if (!res.ok) {
//         throw new Error(result.message || 'Failed to update profile.');
//       }

//       const updatedUser: ProfileUser = result.data;

//       setProfile(updatedUser);
//       setNewImageFile(null);
//       setImagePreview(null);

//       await update({
//         ...session,
//         user: {
//           ...session?.user,
//           name: updatedUser.name,
//           username: updatedUser.username,
//           image: updatedUser.profileImage,
//         },
//       });

//       toast.success("Profile updated successfully!");
//       setIsEditing(false);
//     } catch (error: any) {
//       console.error(error);
//       toast.error(error.message || "An error occurred while saving.");
//     } finally {
//       setIsSaving(false);
//     }
//   };

//   const handleCancel = () => {
//     if(profile) {
//       setFormData({
//         name: profile.name,
//         username: profile.username,
//         bio: profile.bio,
//         interests: profile.interests,
//       });
//     }
//     setNewImageFile(null);
//     setImagePreview(null);
//     setIsEditing(false);
//   }

//   if (status === "loading" || isLoading) {
//     return <div className="flex justify-center items-center h-screen"><Loader2 className="h-8 w-8 animate-spin" /></div>;
//   }
//   if (!profile) {
//     return <div className="text-center mt-20 text-xl">Could not load profile. Please try again later.</div>;
//   }

//   const displayImage = imagePreview || profile.profileImage || "/placeholder.png";

//   return (
//     <div className="max-w-4xl mx-auto p-4 md:p-6">
//       <Card>
//         <CardContent className="p-6">
//           <div className="flex flex-col sm:flex-row items-center gap-6">
//             <div className="relative">
//               <img
//                 src={displayImage}
//                 alt="Profile"
//                 className="w-24 h-24 md:w-32 md:h-32 rounded-full object-cover border-4 border-primary"
//               />
//               {isEditing && (
//                 <>
//                   <input
//                     type="file"
//                     ref={fileInputRef}
//                     onChange={handleFileChange}
//                     accept="image/png, image/jpeg, image/gif"
//                     className="hidden"
//                   />
//                   <Button
//                     variant="outline"
//                     size="icon"
//                     className="absolute bottom-0 right-0 rounded-full"
//                     onClick={() => fileInputRef.current?.click()}
//                   >
//                     <Camera className="h-4 w-4" />
//                   </Button>
//                 </>
//               )}
//             </div>

//             <div className="flex-1 text-center sm:text-left">
//               {isEditing ? (
//                 <div className="space-y-2">
//                    <Input value={formData.name} onChange={(e) => setFormData({ ...formData, name: e.target.value })} className="text-xl font-bold" placeholder="Full Name" />
//                    <Input value={formData.username} onChange={(e) => setFormData({ ...formData, username: e.target.value })} placeholder="username" />
//                 </div>
//               ) : (
//                 <>
//                   <h1 className="text-2xl md:text-3xl font-bold">{profile.name}</h1>
//                   <p className="text-muted-foreground">@{profile.username}</p>
//                 </>
//               )}
//               <p className="text-muted-foreground mt-1">{profile.email}</p>
//               <div className="flex justify-center sm:justify-start gap-4 mt-3">
//                 <div className="text-center"><span className="font-bold">{profile.followers.length}</span><p className="text-sm text-muted-foreground">Followers</p></div>
//                 <div className="text-center"><span className="font-bold">{profile.following.length}</span><p className="text-sm text-muted-foreground">Following</p></div>
//               </div>
//             </div>

//             <div className="flex flex-col sm:flex-row gap-2 self-start">
//             {isEditing ? (
//               <>
//                  <Button onClick={handleSave} disabled={isSaving}>
//                   {isSaving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Save className="mr-2 h-4 w-4"/>}
//                   Save
//                 </Button>
//                 <Button variant="outline" onClick={handleCancel}>Cancel</Button>
//               </>
//             ) : (
//               <Button variant="outline" onClick={() => setIsEditing(true)}>
//                 <Edit3 className="mr-2 h-4 w-4" /> Edit Profile
//               </Button>
//             )}
//             </div>
//           </div>

//           <div className="mt-6 space-y-4">
//             <div>
//               <h3 className="font-semibold text-lg">Bio</h3>
//               {isEditing ? (
//                 <Textarea value={formData.bio} onChange={(e) => setFormData({ ...formData, bio: e.target.value })} placeholder="Tell us about yourself..." />
//               ) : (
//                 <p className="text-muted-foreground">{profile.bio || "No bio added yet."}</p>
//               )}
//             </div>

//             <div>
//               <h3 className="font-semibold text-lg mb-2">Interests</h3>
//               <div className="flex flex-wrap gap-3">
//                 {isEditing ? (
//                   availableInterests.map((interest) => (
//                     <div key={interest} className="flex items-center gap-2 p-2 border rounded-lg">
//                       <Switch id={interest} checked={formData.interests.includes(interest)} onCheckedChange={() => handleInterestToggle(interest)} />
//                       <label htmlFor={interest} className="text-sm font-medium">{interest}</label>
//                     </div>
//                   ))
//                 ) : (
//                   profile.interests.length > 0 ? (
//                     profile.interests.map((interest) => (<Badge key={interest} variant="secondary" className="text-base">{interest}</Badge>))
//                   ) : <p className="text-sm text-muted-foreground">No interests added.</p>
//                 )}
//               </div>
//             </div>
//           </div>
//         </CardContent>
//       </Card>
//     </div>
//   );
// }

// 'use client'

// import { useSession } from "next-auth/react";
// import { useEffect, useState, useRef } from "react";
// import { toast } from "sonner";
// import { upload } from '@imagekit/next';

// import { Card, CardContent } from "@/components/ui/card";
// import { Button } from "@/components/ui/button";
// import { Badge } from "@/components/ui/badge";
// import { Switch } from "@/components/ui/switch";
// import { Input } from "@/components/ui/input";
// import { Textarea } from "@/components/ui/textarea";
// import { Loader2, Edit3, Save, Camera } from "lucide-react";

// // Define the interface for a single Post
// interface Post {
//   _id: string;
//   content: string; // The text of the post
//   photo?: string;   // The URL for an image
//   video?: string;   // The URL for a video
//   createdAt: string;
// }

// // Define the interface for the full User Profile, including their posts
// interface ProfileUser {
//   _id: string;
//   name: string;
//   username: string;
//   email: string;
//   bio: string;
//   profileImage: string;
//   interests: string[];
//   posts: Post[];
//   followers: string[];
//   following: string[];
// }

// export default function ProfilePage() {
//   const { data: session, status, update } = useSession();
//   const [profile, setProfile] = useState<ProfileUser | null>(null);
//   const [isLoading, setIsLoading] = useState(true);
//   const [isEditing, setIsEditing] = useState(false);
//   const [isSaving, setIsSaving] = useState(false);
//   const fileInputRef = useRef<HTMLInputElement>(null);

//   const [newImageFile, setNewImageFile] = useState<File | null>(null);
//   const [imagePreview, setImagePreview] = useState<string | null>(null);

//   const [formData, setFormData] = useState({
//     name: "",
//     username: "",
//     bio: "",
//     interests: [] as string[],
//   });

//   const availableInterests = ["Coding", "Design", "AI", "Music", "Art", "Gaming", "Sports"];

//   // Fetch the full user profile, including posts, from your backend
//   useEffect(() => {
//     const fetchProfile = async () => {
//       if (status === "authenticated" && session?.user?.id) {
//         setIsLoading(true);
//         try {
//           const res = await fetch(`/api/users/${session.user.id}`);
//           if (!res.ok) throw new Error("Failed to fetch profile");
//           const data: ProfileUser = await res.json();
//           setProfile(data);
//           setFormData({
//             name: data.name,
//             username: data.username,
//             bio: data.bio || "",
//             interests: data.interests || [],
//           });
//         } catch (error) {
//           console.error(error);
//           toast.error("Could not load your profile.");
//         } finally {
//           setIsLoading(false);
//         }
//       }
//     };
//     fetchProfile();
//   }, [status, session?.user?.id]);

//   const handleInterestToggle = (interest: string) => {
//     setFormData((prev) => ({
//       ...prev,
//       interests: prev.interests.includes(interest)
//         ? prev.interests.filter((i) => i !== interest)
//         : [...prev.interests, interest],
//     }));
//   };

//   const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
//     const file = e.target.files?.[0];
//     if (file) {
//       setNewImageFile(file);
//       setImagePreview(URL.createObjectURL(file));
//     }
//   };

//   const handleImageUpload = async () => {
//     if (!newImageFile) return '';
//     try {
//       const authRes = await fetch('/api/imagekit-auth');
//       if (!authRes.ok) throw new Error("Failed to authenticate with ImageKit");
//       const authParams = await authRes.json();
//       const response = await upload({
//         file: newImageFile,
//         fileName: newImageFile.name,
//         ...authParams,
//       });
//       return response.url;
//     } catch (error) {
//       console.error("Image upload failed:", error);
//       toast.error("Image upload failed. Please try again.");
//       return '';
//     }
//   };

//   const handleSave = async () => {
//     if (!profile) return;
//     setIsSaving(true);
//     try {
//       let updatedProfileImageUrl = profile.profileImage;
//       if (newImageFile) {
//         const uploadedUrl = await handleImageUpload();
//         if (uploadedUrl) {
//           updatedProfileImageUrl = uploadedUrl;
//         } else {
//           setIsSaving(false);
//           return;
//         }
//       }
//       const finalFormData = {
//         ...formData,
//         profileImage: updatedProfileImageUrl,
//       };
//       const res = await fetch(`/api/users/${profile._id}`, {
//         method: 'PUT',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(finalFormData),
//       });
//       const result = await res.json();
//       if (!res.ok) {
//         throw new Error(result.message || 'Failed to update profile.');
//       }
//       const updatedUser: ProfileUser = result.data;
//       setProfile(updatedUser);
//       setNewImageFile(null);
//       setImagePreview(null);
//       await update({
//         ...session,
//         user: {
//           ...session?.user,
//           name: updatedUser.name,
//           username: updatedUser.username,
//           image: updatedUser.profileImage,
//         },
//       });
//       toast.success("Profile updated successfully!");
//       setIsEditing(false);
//     } catch (error: any) {
//       console.error(error);
//       toast.error(error.message || "An error occurred while saving.");
//     } finally {
//       setIsSaving(false);
//     }
//   };

//   const handleCancel = () => {
//     if (profile) {
//       setFormData({
//         name: profile.name,
//         username: profile.username,
//         bio: profile.bio,
//         interests: profile.interests,
//       });
//     }
//     setNewImageFile(null);
//     setImagePreview(null);
//     setIsEditing(false);
//   };

//   if (status === "loading" || isLoading) {
//     return (
//       <div className="flex justify-center items-center h-screen">
//         <Loader2 className="h-8 w-8 animate-spin" />
//       </div>
//     );
//   }

//   if (!profile) {
//     return (
//       <div className="text-center mt-20 text-xl">
//         Could not load profile. Please try again later.
//       </div>
//     );
//   }

//   const displayImage = imagePreview || profile.profileImage || "/placeholder.png";

//   return (
//     <div className="max-w-4xl mx-auto p-4 md:p-6 space-y-8">
//       <Card>
//         <CardContent className="p-6">
//           <div className="flex flex-col sm:flex-row items-center gap-6">
//             <div className="relative">
//               <img
//                 src={displayImage}
//                 alt="Profile"
//                 className="w-24 h-24 md:w-32 md:h-32 rounded-full object-cover border-4 border-primary"
//               />
//               {isEditing && (
//                 <>
//                   <input
//                     type="file"
//                     ref={fileInputRef}
//                     onChange={handleFileChange}
//                     accept="image/*"
//                     className="hidden"
//                   />
//                   <Button
//                     variant="outline"
//                     size="icon"
//                     className="absolute bottom-0 right-0 rounded-full"
//                     onClick={() => fileInputRef.current?.click()}
//                   >
//                     <Camera className="h-4 w-4" />
//                   </Button>
//                 </>
//               )}
//             </div>

//             <div className="flex-1 text-center sm:text-left">
//               {isEditing ? (
//                 <div className="space-y-2">
//                   <Input
//                     value={formData.name}
//                     onChange={(e) => setFormData({ ...formData, name: e.target.value })}
//                     className="text-xl font-bold"
//                     placeholder="Full Name"
//                   />
//                   <Input
//                     value={formData.username}
//                     onChange={(e) => setFormData({ ...formData, username: e.target.value })}
//                     placeholder="username"
//                   />
//                 </div>
//               ) : (
//                 <>
//                   <h1 className="text-2xl md:text-3xl font-bold">{profile.name}</h1>
//                   <p className="text-muted-foreground">@{profile.username}</p>
//                 </>
//               )}
//               <p className="text-muted-foreground mt-1">{profile.email}</p>
//               <div className="flex justify-center sm:justify-start gap-4 mt-3">
//                 <div className="text-center">
//                   <span className="font-bold">{profile.posts.length}</span>
//                   <p className="text-sm text-muted-foreground">Posts</p>
//                 </div>
//                 <div className="text-center">
//                   <span className="font-bold">{profile.followers.length}</span>
//                   <p className="text-sm text-muted-foreground">Followers</p>
//                 </div>
//                 <div className="text-center">
//                   <span className="font-bold">{profile.following.length}</span>
//                   <p className="text-sm text-muted-foreground">Following</p>
//                 </div>
//               </div>
//             </div>

//             <div className="flex flex-col sm:flex-row gap-2 self-start">
//               {isEditing ? (
//                 <>
//                   <Button onClick={handleSave} disabled={isSaving}>
//                     {isSaving ? (
//                       <Loader2 className="mr-2 h-4 w-4 animate-spin" />
//                     ) : (
//                       <Save className="mr-2 h-4 w-4" />
//                     )}
//                     Save
//                   </Button>
//                   <Button variant="outline" onClick={handleCancel}>
//                     Cancel
//                   </Button>
//                 </>
//               ) : (
//                 <Button variant="outline" onClick={() => setIsEditing(true)}>
//                   <Edit3 className="mr-2 h-4 w-4" /> Edit Profile
//                 </Button>
//               )}
//             </div>
//           </div>

//           <div className="mt-6 space-y-4">
//             <div>
//               <h3 className="font-semibold text-lg">Bio</h3>
//               {isEditing ? (
//                 <Textarea
//                   value={formData.bio}
//                   onChange={(e) => setFormData({ ...formData, bio: e.target.value })}
//                   placeholder="Tell us about yourself..."
//                 />
//               ) : (
//                 <p className="text-muted-foreground">
//                   {profile.bio || "No bio added yet."}
//                 </p>
//               )}
//             </div>
//             <div>
//               <h3 className="font-semibold text-lg mb-2">Interests</h3>
//               <div className="flex flex-wrap gap-3">
//                 {isEditing ? (
//                   availableInterests.map((interest) => (
//                     <div
//                       key={interest}
//                       className="flex items-center gap-2 p-2 border rounded-lg"
//                     >
//                       <Switch
//                         id={interest}
//                         checked={formData.interests.includes(interest)}
//                         onCheckedChange={() => handleInterestToggle(interest)}
//                       />
//                       <label htmlFor={interest} className="text-sm font-medium">
//                         {interest}
//                       </label>
//                     </div>
//                   ))
//                 ) : profile.interests.length > 0 ? (
//                   profile.interests.map((interest) => (
//                     <Badge key={interest} variant="secondary" className="text-base">
//                       {interest}
//                     </Badge>
//                   ))
//                 ) : (
//                   <p className="text-sm text-muted-foreground">
//                     No interests added.
//                   </p>
//                 )}
//               </div>
//             </div>
//           </div>
//         </CardContent>
//       </Card>

//       {/* Section for displaying user's posts */}
//       <div>
//         <h2 className="text-2xl font-bold">Posts</h2>
//         {profile.posts.length > 0 ? (
//           <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mt-4">
//             {profile.posts.map((post) => (
//               <Card key={post._id} className="overflow-hidden group">
//                 {/* Display photo if it exists */}
//                 {post.photo && (
//                   <div className="aspect-square w-full overflow-hidden">
//                     <img
//                       src={post.photo}
//                       alt="Post content"
//                       className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
//                     />
//                   </div>
//                 )}
//                 {/* You can add similar logic for video if needed */}
//                 <CardContent className="p-4">
//                   <p className="text-sm truncate" title={post.content}>
//                     {post.content}
//                   </p>
//                   <p className="text-xs text-muted-foreground mt-2">
//                     {new Date(post.createdAt).toLocaleDateString("en-IN", {
//                       day: 'numeric',
//                       month: 'long',
//                       year: 'numeric',
//                     })}
//                   </p>
//                 </CardContent>
//               </Card>
//             ))}
//           </div>
//         ) : (
//           <div className="text-center py-10 border-2 border-dashed rounded-lg mt-4">
//             <p className="text-muted-foreground">
//               This user hasn&apost posted anything yet.
//             </p>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// }

// 'use client'

// import { useSession } from "next-auth/react";
// import { useEffect, useState, useRef, useMemo } from "react";
// import { toast } from "sonner";
// import { upload } from '@imagekit/next';

// import { Card, CardContent } from "@/components/ui/card";
// import { Button } from "@/components/ui/button";
// import { Badge } from "@/components/ui/badge";
// import { Switch } from "@/components/ui/switch";
// import { Input } from "@/components/ui/input";
// import { Textarea } from "@/components/ui/textarea";
// import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
// import { Loader2, Edit3, Save, Camera, Image as ImageIcon, Video, FileText } from "lucide-react";

// // --- TYPE INTERFACES ---

// interface Post {
//   _id: string;
//   text: string;
//   photo?: string;
//   video?: string;
//   createdAt: string;
// }

// interface ProfileUser {
//   _id: string;
//   name: string;
//   username: string;
//   email: string;
//   bio: string;
//   profileImage: string;
//   interests: string[];
//   posts: Post[];
//   followers: string[];
//   following: string[];
// }


// // --- HELPER COMPONENT for rendering individual posts ---

// // const PostCard = ({ post }: { post: Post }) => {
// //   const postType = post.photo ? 'photo' : post.video ? 'video' : 'text';

// //   // Text-only posts have a different, more prominent layout
// //   if (postType === 'text') {
// //     return (
// //       <Card key={post._id} className="col-span-1 sm:col-span-2 md:col-span-3">
// //         <CardContent className="p-6">
// //           <p className="text-md whitespace-pre-wrap">{post.text}</p>
// //           <p className="text-xs text-muted-foreground mt-4">
// //             {new Date(post.createdAt).toLocaleDateString("en-IN", {
// //               day: 'numeric', month: 'long', year: 'numeric'
// //             })}
// //           </p>
// //         </CardContent>
// //       </Card>
// //     );
// //   }


// const PostCard = ({ post }: { post: Post }) => {
//   const postType = post.photo ? 'photo' : post.video ? 'video' : 'text';

//   // Text-only posts have a different, more prominent layout
//   if (postType === 'text') {
//     return (
//       // ✅ CHANGE on this line: md:col-span-3 -> md:col-span-2
//       <Card key={post._id} className="col-span-1 sm:col-span-2 md:col-span-2"> 
//         <CardContent className="p-6">
//           <p className="text-md whitespace-pre-wrap">{post.text}</p>
//           <p className="text-xs text-muted-foreground mt-4">
//             {new Date(post.createdAt).toLocaleDateString("en-IN", {
//               day: 'numeric', month: 'long', year: 'numeric'
//             })}
//           </p>
//         </CardContent>
//       </Card>
//     );
//   }

//   // Media posts (photo/video) are displayed as a grid item
//   return (
//     <Card key={post._id} className="overflow-hidden group relative">
//       <div className="absolute top-2 right-2 bg-black/50 p-1.5 rounded-full z-10">
//         {postType === 'photo' && <ImageIcon className="w-4 h-4 text-white" />}
//         {postType === 'video' && <Video className="w-4 h-4 text-white" />}
//       </div>

//       <div className="aspect-square w-full overflow-hidden bg-gray-100">
//         {post.photo && (
//           <img
//             src={post.photo}
//             alt="Post content"
//             className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
//           />
//         )}
//         {post.video && (
//           <div className="w-full h-full flex items-center justify-center">
//             <Video className="w-16 h-16 text-gray-400" />
//           </div>
//         )}
//       </div>
//       {post.text && (
//          <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center p-4">
//             <p className="text-white text-center text-sm">{post.text}</p>
//          </div>
//       )}
//     </Card>
//   );
// };


// // --- MAIN PROFILE PAGE COMPONENT ---

// export default function ProfilePage() {
//   const { data: session, status, update } = useSession();
//   const [profile, setProfile] = useState<ProfileUser | null>(null);
//   const [activeTab, setActiveTab] = useState('all');

//   const [isLoading, setIsLoading] = useState(true);
//   const [isEditing, setIsEditing] = useState(false);
//   const [isSaving, setIsSaving] = useState(false);
//   const fileInputRef = useRef<HTMLInputElement>(null);
//   const [newImageFile, setNewImageFile] = useState<File | null>(null);
//   const [imagePreview, setImagePreview] = useState<string | null>(null);
//   const [formData, setFormData] = useState({ name: "", username: "", bio: "", interests: [] as string[] });

//   const availableInterests = ["Coding", "Design", "AI", "Music", "Art", "Gaming", "Sports"];

//   useEffect(() => {
//     const fetchProfile = async () => {
//       if (status === "authenticated" && session?.user?.id) {
//         setIsLoading(true);
//         try {
//           const res = await fetch(`/api/users/${session.user.id}`);
//           if (!res.ok) throw new Error("Failed to fetch profile");
//           const data: ProfileUser = await res.json();
//           setProfile(data);
//           setFormData({ name: data.name, username: data.username, bio: data.bio || "", interests: data.interests || [] });
//         } catch (error) {
//           console.error(error);
//           toast.error("Could not load your profile.");
//         } finally {
//           setIsLoading(false);
//         }
//       }
//     };
//     fetchProfile();
//   }, [status, session?.user?.id]);

//   const filteredPosts = useMemo(() => {
//     if (!profile) return [];
//     switch (activeTab) {
//       case 'photos':
//         return profile.posts.filter(p => !!p.photo);
//       case 'videos':
//         return profile.posts.filter(p => !!p.video);
//       case 'text':
//         return profile.posts.filter(p => !p.photo && !p.video);
//       case 'all':
//       default:
//         return profile.posts;
//     }
//   }, [profile, activeTab]);

//   const handleInterestToggle = (interest: string) => { setFormData((prev) => ({...prev, interests: prev.interests.includes(interest) ? prev.interests.filter((i) => i !== interest) : [...prev.interests, interest]})); };
//   const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => { const file = e.target.files?.[0]; if (file) { setNewImageFile(file); setImagePreview(URL.createObjectURL(file)); }};
//   const handleImageUpload = async () => { if (!newImageFile) return ''; try { const authRes = await fetch('/api/imagekit-auth'); if (!authRes.ok) throw new Error("Failed to authenticate with ImageKit"); const authParams = await authRes.json(); const response = await upload({ file: newImageFile, fileName: newImageFile.name, ...authParams, }); return response.url; } catch (error) { console.error("Image upload failed:", error); toast.error("Image upload failed. Please try again."); return ''; }};
//   const handleSave = async () => { if (!profile) return; setIsSaving(true); try { let updatedProfileImageUrl = profile.profileImage; if (newImageFile) { const uploadedUrl = await handleImageUpload(); if (uploadedUrl) { updatedProfileImageUrl = uploadedUrl; } else { setIsSaving(false); return; } } const finalFormData = { ...formData, profileImage: updatedProfileImageUrl, }; const res = await fetch(`/api/users/${profile._id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(finalFormData), }); const result = await res.json(); if (!res.ok) { throw new Error(result.message || 'Failed to update profile.'); } const updatedUser: ProfileUser = result.data; setProfile(updatedUser); setNewImageFile(null); setImagePreview(null); await update({ ...session, user: { ...session?.user, name: updatedUser.name, username: updatedUser.username, image: updatedUser.profileImage, }, }); toast.success("Profile updated successfully!"); setIsEditing(false); } catch (error: any) { console.error(error); toast.error(error.message || "An error occurred while saving."); } finally { setIsSaving(false); }};
//   const handleCancel = () => { if(profile) { setFormData({ name: profile.name, username: profile.username, bio: profile.bio, interests: profile.interests, }); } setNewImageFile(null); setImagePreview(null); setIsEditing(false); };

//   if (status === "loading" || isLoading) {
//     return <div className="flex justify-center items-center h-screen"><Loader2 className="h-8 w-8 animate-spin" /></div>;
//   }
//   if (!profile) {
//     return <div className="text-center mt-20 text-xl">Could not load profile. Please try again later.</div>;
//   }

//   const displayImage = imagePreview || profile.profileImage || "/placeholder.png";

//   return (
//     <div className="max-w-4xl mx-auto p-4 md:p-6 space-y-8">
//       {/* --- PROFILE CARD SECTION --- */}
//       <Card>
//         <CardContent className="p-6">
//           <div className="flex flex-col sm:flex-row items-center gap-6">
//             <div className="relative">
//               <img src={displayImage} alt="Profile" className="w-24 h-24 md:w-32 md:h-32 rounded-full object-cover border-4 border-primary"/>
//               {isEditing && (<> <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" className="hidden"/> <Button variant="outline" size="icon" className="absolute bottom-0 right-0 rounded-full" onClick={() => fileInputRef.current?.click()}> <Camera className="h-4 w-4" /> </Button> </>)}
//             </div>

//             <div className="flex-1 text-center sm:text-left">
//               {isEditing ? ( <div className="space-y-2"> <Input value={formData.name} onChange={(e) => setFormData({ ...formData, name: e.target.value })} className="text-xl font-bold" placeholder="Full Name" /> <Input value={formData.username} onChange={(e) => setFormData({ ...formData, username: e.target.value })} placeholder="username" /> </div> ) : ( <> <h1 className="text-2xl md:text-3xl font-bold">{profile.name}</h1> <p className="text-muted-foreground">@{profile.username}</p> </> )}
//               <p className="text-muted-foreground mt-1">{profile.email}</p>
//               <div className="flex justify-center sm:justify-start gap-4 mt-3">
//                 <div className="text-center"><span className="font-bold">{profile.posts.length}</span><p className="text-sm text-muted-foreground">Posts</p></div>
//                 <div className="text-center"><span className="font-bold">{profile.followers.length}</span><p className="text-sm text-muted-foreground">Followers</p></div>
//                 <div className="text-center"><span className="font-bold">{profile.following.length}</span><p className="text-sm text-muted-foreground">Following</p></div>
//               </div>
//             </div>

//             <div className="flex flex-col sm:flex-row gap-2 self-start">
//               {isEditing ? ( <> <Button onClick={handleSave} disabled={isSaving}> {isSaving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Save className="mr-2 h-4 w-4"/>} Save </Button> <Button variant="outline" onClick={handleCancel}>Cancel</Button> </> ) : ( <Button variant="outline" onClick={() => setIsEditing(true)}> <Edit3 className="mr-2 h-4 w-4" /> Edit Profile </Button> )}
//             </div>
//           </div>
//           <div className="mt-6 space-y-4">
//             <div> <h3 className="font-semibold text-lg">Bio</h3> {isEditing ? ( <Textarea value={formData.bio} onChange={(e) => setFormData({ ...formData, bio: e.target.value })} placeholder="Tell us about yourself..." /> ) : ( <p className="text-muted-foreground">{profile.bio || "No bio added yet."}</p> )} </div>
//             <div> <h3 className="font-semibold text-lg mb-2">Interests</h3> <div className="flex flex-wrap gap-3"> {isEditing ? ( availableInterests.map((interest) => ( <div key={interest} className="flex items-center gap-2 p-2 border rounded-lg"> <Switch id={interest} checked={formData.interests.includes(interest)} onCheckedChange={() => handleInterestToggle(interest)} /> <label htmlFor={interest} className="text-sm font-medium">{interest}</label> </div> )) ) : ( profile.interests.length > 0 ? ( profile.interests.map((interest) => (<Badge key={interest} variant="secondary" className="text-base">{interest}</Badge>)) ) : <p className="text-sm text-muted-foreground">No interests added.</p> )} </div> </div>
//           </div>
//         </CardContent>
//       </Card>

//       {/* --- POSTS SECTION with TABS --- */}
//       <div>
//         <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
//           <TabsList className="grid w-full grid-cols-4">
//             <TabsTrigger value="all">All</TabsTrigger>
//             <TabsTrigger value="photos" className="flex items-center gap-2"><ImageIcon className="w-4 h-4" />Photos</TabsTrigger>
//             <TabsTrigger value="videos" className="flex items-center gap-2"><Video className="w-4 h-4" />Videos</TabsTrigger>
//             <TabsTrigger value="text" className="flex items-center gap-2"><FileText className="w-4 h-4" />Text</TabsTrigger>
//           </TabsList>

//           <div className="mt-4">
//             {filteredPosts.length > 0 ? (
//               <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
//                 {filteredPosts.map((post) => (
//                   <PostCard key={post._id} post={post} />
//                 ))}
//               </div>
//             ) : (
//               <div className="text-center py-16 border-2 border-dashed rounded-lg mt-4">
//                 <p className="text-muted-foreground">No posts in this category yet.</p>
//               </div>
//             )}
//           </div>
//         </Tabs>
//       </div>
//     </div>
//   );
// }


'use client'

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
import { Loader2, Edit3, Save, Camera, Image as ImageIcon, Video, FileText } from "lucide-react";
import { PostDetailModal } from '@/components/posts/PostDetailModal'; // Import the modal component

// --- TYPE INTERFACES ---

interface Post {
  _id: string;
  text: string;
  photo?: string;
  video?: string;
  createdAt: string;
  // Add other fields needed for the modal, assuming they are populated by the API
  author: any;
  likes: string[];
  savedBy: string[];
}

interface ProfileUser {
  _id: string;
  name: string;
  username: string;
  email: string;
  bio: string;
  profileImage: string;
  interests: string[];
  posts: Post[];
  followers: string[];
  following: string[];
}

// --- HELPER COMPONENT for rendering individual posts in the grid ---

const PostCard = ({ post, onClick }: { post: Post, onClick: () => void }) => {
  const postType = post.photo ? 'photo' : post.video ? 'video' : 'text';

  if (postType === 'text') {
    return (
      <Card onClick={onClick} key={post._id} className="col-span-1 sm:col-span-2 md:col-span-2 cursor-pointer hover:bg-gray-50 transition">
        <CardContent className="p-6">
          <p className="text-md whitespace-pre-wrap">{post.text}</p>
          <p className="text-xs text-muted-foreground mt-4">
            {new Date(post.createdAt).toLocaleDateString("en-IN", {
              day: 'numeric', month: 'long', year: 'numeric'
            })}
          </p>
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
        {post.photo && <img src={post.photo} alt="Post content" className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />}
        {post.video && <div className="w-full h-full flex items-center justify-center"><Video className="w-16 h-16 text-gray-400" /></div>}
      </div>
      {post.text && <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center p-4"><p className="text-white text-center text-sm">{post.text}</p></div>}
    </Card>
  );
};

// --- MAIN PROFILE PAGE COMPONENT ---

export default function ProfilePage() {
  const { data: session, status, update } = useSession();
  const [profile, setProfile] = useState<ProfileUser | null>(null);
  const [activeTab, setActiveTab] = useState('all');

  // State for the Post Detail Modal
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedPostIndex, setSelectedPostIndex] = useState(0);

  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [newImageFile, setNewImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [formData, setFormData] = useState({ name: "", username: "", bio: "", interests: [] as string[] });

  const availableInterests = ["Coding", "Design", "AI", "Music", "Art", "Gaming", "Sports"];

  useEffect(() => {
    const fetchProfile = async () => {
      if (status === "authenticated" && session?.user?.id) {
        setIsLoading(true);
        try {
          const res = await fetch(`/api/users/${session.user.id}`);
          if (!res.ok) throw new Error("Failed to fetch profile");
          const data: ProfileUser = await res.json();
          setProfile(data);
          setFormData({ name: data.name, username: data.username, bio: data.bio || "", interests: data.interests || [] });
        } catch (error) {
          console.error(error);
          toast.error("Could not load your profile.");
        } finally {
          setIsLoading(false);
        }
      }
    };
    fetchProfile();
  }, [status, session?.user?.id]);

  const filteredPosts = useMemo(() => {
    if (!profile) return [];
    switch (activeTab) {
      case 'photos': return profile.posts.filter(p => !!p.photo);
      case 'videos': return profile.posts.filter(p => !!p.video);
      case 'text': return profile.posts.filter(p => !p.photo && !p.video);
      default: return profile.posts;
    }
  }, [profile, activeTab]);

  const openPostModal = (index: number) => {
    setSelectedPostIndex(index);
    setIsModalOpen(true);
  };


  // ✅ 1. DEFINE THE HANDLER FUNCTION
  // This function allows the modal to update the main profile state
  const handlePostUpdate = (updatedPost: Post) => {
    setProfile(prevProfile => {
      if (!prevProfile) return null;

      const newPosts = prevProfile.posts.map(post =>
        post._id === updatedPost._id ? updatedPost : post
      );

      return { ...prevProfile, posts: newPosts };
    });
  };


  const handleInterestToggle = (interest: string) => { setFormData((prev) => ({ ...prev, interests: prev.interests.includes(interest) ? prev.interests.filter((i) => i !== interest) : [...prev.interests, interest] })); };
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => { const file = e.target.files?.[0]; if (file) { setNewImageFile(file); setImagePreview(URL.createObjectURL(file)); } };
  const handleImageUpload = async () => { if (!newImageFile) return ''; try { const authRes = await fetch('/api/imagekit-auth'); if (!authRes.ok) throw new Error("Failed to authenticate with ImageKit"); const authParams = await authRes.json(); const response = await upload({ file: newImageFile, fileName: newImageFile.name, ...authParams, }); return response.url; } catch (error) { console.error("Image upload failed:", error); toast.error("Image upload failed. Please try again."); return ''; } };
  const handleSave = async () => { if (!profile) return; setIsSaving(true); try { let updatedProfileImageUrl = profile.profileImage; if (newImageFile) { const uploadedUrl = await handleImageUpload(); if (uploadedUrl) { updatedProfileImageUrl = uploadedUrl; } else { setIsSaving(false); return; } } const finalFormData = { ...formData, profileImage: updatedProfileImageUrl, }; const res = await fetch(`/api/users/${profile._id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(finalFormData), }); const result = await res.json(); if (!res.ok) { throw new Error(result.message || 'Failed to update profile.'); } const updatedUser: ProfileUser = result.data; setProfile(updatedUser); setNewImageFile(null); setImagePreview(null); await update({ ...session, user: { ...session?.user, name: updatedUser.name, username: updatedUser.username, image: updatedUser.profileImage, }, }); toast.success("Profile updated successfully!"); setIsEditing(false); } catch (error: any) { console.error(error); toast.error(error.message || "An error occurred while saving."); } finally { setIsSaving(false); } };
  const handleCancel = () => { if (profile) { setFormData({ name: profile.name, username: profile.username, bio: profile.bio, interests: profile.interests, }); } setNewImageFile(null); setImagePreview(null); setIsEditing(false); };

  if (status === "loading" || isLoading) {
    return <div className="flex justify-center items-center h-screen"><Loader2 className="h-8 w-8 animate-spin" /></div>;
  }
  if (!profile) {
    return <div className="text-center mt-20 text-xl">Could not load profile. Please try again later.</div>;
  }

  const displayImage = imagePreview || profile.profileImage || "/placeholder.png";

  return (
    <div className="max-w-4xl mx-auto p-4 md:p-6 space-y-8">
      {/* --- PROFILE CARD SECTION --- */}
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col sm:flex-row items-center gap-6">
            <div className="relative">
              <img src={displayImage} alt="Profile" className="w-24 h-24 md:w-32 md:h-32 rounded-full object-cover border-4 border-primary" />
              {isEditing && (<> <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" className="hidden" /> <Button variant="outline" size="icon" className="absolute bottom-0 right-0 rounded-full" onClick={() => fileInputRef.current?.click()}> <Camera className="h-4 w-4" /> </Button> </>)}
            </div>
            <div className="flex-1 text-center sm:text-left">
              {isEditing ? (<div className="space-y-2"> <Input value={formData.name} onChange={(e) => setFormData({ ...formData, name: e.target.value })} className="text-xl font-bold" placeholder="Full Name" /> <Input value={formData.username} onChange={(e) => setFormData({ ...formData, username: e.target.value })} placeholder="username" /> </div>) : (<> <h1 className="text-2xl md:text-3xl font-bold">{profile.name}</h1> <p className="text-muted-foreground">@{profile.username}</p> </>)}
              <p className="text-muted-foreground mt-1">{profile.email}</p>
              <div className="flex justify-center sm:justify-start gap-4 mt-3">
                <div className="text-center"><span className="font-bold">{profile.posts.length}</span><p className="text-sm text-muted-foreground">Posts</p></div>
                <div className="text-center"><span className="font-bold">{profile.followers.length}</span><p className="text-sm text-muted-foreground">Followers</p></div>
                <div className="text-center"><span className="font-bold">{profile.following.length}</span><p className="text-sm text-muted-foreground">Following</p></div>
              </div>
            </div>
            <div className="flex flex-col sm:flex-row gap-2 self-start">
              {isEditing ? (<> <Button onClick={handleSave} disabled={isSaving}> {isSaving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Save className="mr-2 h-4 w-4" />} Save </Button> <Button variant="outline" onClick={handleCancel}>Cancel</Button> </>) : (<Button variant="outline" onClick={() => setIsEditing(true)}> <Edit3 className="mr-2 h-4 w-4" /> Edit Profile </Button>)}
            </div>
          </div>
          <div className="mt-6 space-y-4">
            <div> <h3 className="font-semibold text-lg">Bio</h3> {isEditing ? (<Textarea value={formData.bio} onChange={(e) => setFormData({ ...formData, bio: e.target.value })} placeholder="Tell us about yourself..." />) : (<p className="text-muted-foreground">{profile.bio || "No bio added yet."}</p>)} </div>
            <div> <h3 className="font-semibold text-lg mb-2">Interests</h3> <div className="flex flex-wrap gap-3"> {isEditing ? (availableInterests.map((interest) => (<div key={interest} className="flex items-center gap-2 p-2 border rounded-lg"> <Switch id={interest} checked={formData.interests.includes(interest)} onCheckedChange={() => handleInterestToggle(interest)} /> <label htmlFor={interest} className="text-sm font-medium">{interest}</label> </div>))) : (profile.posts.length > 0 ? (profile.interests.map((interest) => (<Badge key={interest} variant="secondary" className="text-base">{interest}</Badge>))) : <p className="text-sm text-muted-foreground">No interests added.</p>)} </div> </div>
          </div>
        </CardContent>
      </Card>

      {/* --- POSTS SECTION with TABS --- */}
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
                <p className="text-muted-foreground">No posts in this category yet.</p>
              </div>
            )}
          </div>
        </Tabs>
      </div>

      {/* --- RENDER THE POST DETAIL MODAL --- */}
      {profile && (
        <PostDetailModal
          posts={filteredPosts}
          startIndex={selectedPostIndex}
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          onPostUpdate={handlePostUpdate}
        />
      )}
    </div>
  );
}