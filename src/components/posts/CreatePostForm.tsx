// 'use client';

// import { useState } from 'react';
// import { useSession } from 'next-auth/react';
// import { toast } from 'sonner';
// import { upload } from '@imagekit/next';
// import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
// import { Button } from '@/components/ui/button';
// import { Textarea } from '@/components/ui/textarea';
// import { Input } from '@/components/ui/input';
// import { Label } from '@/components/ui/label';
// import { Loader2, Send, Image as ImageIcon } from 'lucide-react';

// interface CreatePostFormProps {
//   onPostCreated?: (post: any) => void; // Optional callback to refresh a feed
// }

// export function CreatePostForm({ onPostCreated }: CreatePostFormProps) {
//   const { data: session } = useSession();
//   const [text, setText] = useState('');
//   const [topics, setTopics] = useState(''); // Comma-separated
//   const [tags, setTags] = useState('');   // Comma-separated
//   const [file, setFile] = useState<File | null>(null);
//   const [isSubmitting, setIsSubmitting] = useState(false);

//   const resetForm = () => {
//     setText('');
//     setTopics('');
//     setTags('');
//     setFile(null);
//   };

//   const handleUpload = async () => {
//     if (!file) return { url: '', fileType: '' };

//     try {
//       const authRes = await fetch('/api/imagekit-auth');
//       if (!authRes.ok) throw new Error('Failed to authenticate with ImageKit');
//       const authParams = await authRes.json();

//       const response = await upload({
//         file,
//         fileName: file.name,
//         ...authParams,
//       });

//       return {
//         url: response.url,
//         fileType: file.type.startsWith('image/') ? 'photo' : 'video',
//       };
//     } catch (error) {
//       console.error('Upload failed:', error);
//       toast.error('Media upload failed. Please try again.');
//       throw error; // Propagate error to stop submission
//     }
//   };

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!session) {
//       toast.error('You must be logged in to post.');
//       return;
//     }
//     if (!text && !file) {
//       toast.error('Your post must include text or media.');
//       return;
//     }

//     setIsSubmitting(true);

//     try {
//       const mediaData = { photo: '', video: '' };
//       if (file) {
//         const { url, fileType } = await handleUpload();
//         if (fileType === 'photo') {
//           mediaData.photo = url;
//         } else {
//           mediaData.video = url;
//         }
//       }

//       const body = {
//         text,
//         ...mediaData,
//         topics: topics.split(',').map(t => t.trim()).filter(Boolean),
//         tags: tags.split(',').map(t => t.trim()).filter(Boolean),
//       };

//       const res = await fetch('/api/posts', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(body),
//       });

//       if (!res.ok) {
//         const errorData = await res.json();
//         throw new Error(errorData.message || 'Failed to create post.');
//       } 

//       const newPost = await res.json();
//       toast.success('Post created successfully!');
//       resetForm();
//       onPostCreated?.(newPost); // Trigger callback if provided

//     } catch (error: any) {
//       toast.error(error.message);
//     } finally {
//       setIsSubmitting(false);
//     }
//   };

//   return (
//     <Card className="w-full max-w-lg mx-auto">
//       <CardHeader>
//         <CardTitle>Create a New Post</CardTitle>
//       </CardHeader>
//       <CardContent>
//         <form onSubmit={handleSubmit} className="space-y-4">
//           <Textarea
//             placeholder="What's on your mind?"
//             value={text}
//             onChange={(e) => setText(e.target.value)}
//             rows={4}
//             disabled={isSubmitting}
//           />
//           <div className="grid grid-cols-2 gap-4">
//             <div>
//               <Label htmlFor="topics">Topics (comma-separated)</Label>
//               <Input id="topics" placeholder="e.g. sports, news" value={topics} onChange={(e) => setTopics(e.target.value)} disabled={isSubmitting} />
//             </div>
//             <div>
//               <Label htmlFor="tags">Tags (comma-separated)</Label>
//               <Input id="tags" placeholder="e.g. fun, travel" value={tags} onChange={(e) => setTags(e.target.value)} disabled={isSubmitting} />
//             </div>
//           </div>
//           <div>
//             <Label htmlFor="media">Add Photo or Video</Label>
//             <Input
//               id="media"
//               type="file"
//               accept="image/*,video/*"
//               onChange={(e) => setFile(e.target.files?.[0] || null)}
//               disabled={isSubmitting}
//             />
//             {file && <p className="text-sm text-muted-foreground mt-2">Selected: {file.name}</p>}
//           </div>
//           <Button type="submit" className="w-full" disabled={isSubmitting}>
//             {isSubmitting ? (
//               <Loader2 className="mr-2 h-4 w-4 animate-spin" />
//             ) : (
//               <Send className="mr-2 h-4 w-4" />
//             )}
//             {isSubmitting ? 'Posting...' : 'Post'}
//           </Button>
//         </form>
//       </CardContent>
//     </Card>
//   );
// }


// 'use client';

// import { useState } from 'react';
// import { useSession } from 'next-auth/react';
// import { toast } from 'sonner';
// import { upload } from '@imagekit/next';
// import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
// import { Button } from '@/components/ui/button';
// import { Textarea } from '@/components/ui/textarea';
// import { Input } from '@/components/ui/input';
// import { Label } from '@/components/ui/label';
// import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"; // Import RadioGroup
// import { Loader2, Send } from 'lucide-react';

// // Define a more specific type for the user's gender in the session
// type Gender = "male" | "female" | "trans" | "other" | "common";

// interface CreatePostFormProps {
//   onPostCreated?: (post: any) => void;
// }

// export function CreatePostForm({ onPostCreated }: CreatePostFormProps) {
//   const { data: session } = useSession();
  
//   // --- STATE ---
//   const [text, setText] = useState('');
//   const [topics, setTopics] = useState('');
//   const [tags, setTags] = useState('');
//   const [file, setFile] = useState<File | null>(null);
//   const [isSubmitting, setIsSubmitting] = useState(false);
//   // ✅ 1. Add state for the selected post space
//   const [space, setSpace] = useState<Gender>('common');

//   // --- DERIVED VALUES ---
//   const userGender = session?.user?.gender as Gender;
//   // Determine which spaces the user is allowed to post in
//   const availableSpaces = 
//     userGender && userGender !== 'common' && userGender !== 'other'
//       ? ['common', userGender]
//       : ['common'];

//   // --- HANDLERS ---
//   const resetForm = () => {
//     setText('');
//     setTopics('');
//     setTags('');
//     setFile(null);
//     setSpace('common'); // Reset space to default
//   };

//   const handleUpload = async () => {
//     if (!file) return { url: '', fileType: '' };
//     try {
//       const authRes = await fetch('/api/imagekit-auth');
//       if (!authRes.ok) throw new Error('Failed to authenticate with ImageKit');
//       const authParams = await authRes.json();
//       const response = await upload({ file, fileName: file.name, ...authParams });
//       return { url: response.url, fileType: file.type.startsWith('image/') ? 'photo' : 'video' };
//     } catch (error) {
//       console.error('Upload failed:', error);
//       toast.error('Media upload failed. Please try again.');
//       throw error;
//     }
//   };

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!session) { toast.error('You must be logged in to post.'); return; }
//     if (!text && !file) { toast.error('Your post must include text or media.'); return; }

//     setIsSubmitting(true);
//     try {
//       const mediaData = { photo: '', video: '' };
//       if (file) {
//         const { url, fileType } = await handleUpload();
//         if (fileType === 'photo') mediaData.photo = url;
//         else mediaData.video = url;
//       }

//       // ✅ 2. Add the selected 'space' to the request body
//       const body = {
//         text,
//         ...mediaData,
//         topics: topics.split(',').map(t => t.trim()).filter(Boolean),
//         tags: tags.split(',').map(t => t.trim()).filter(Boolean),
//         space, // Add the selected space
//       };

//       const res = await fetch('/api/posts', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(body),
//       });

//       if (!res.ok) {
//         const errorData = await res.json();
//         throw new Error(errorData.message || 'Failed to create post.');
//       } 

//       const newPost = await res.json();
//       toast.success(`Post created in '${space}' space!`);
//       resetForm();
//       onPostCreated?.(newPost);

//     } catch (error: any) {
//       toast.error(error.message);
//     } finally {
//       setIsSubmitting(false);
//     }
//   };

//   return (
//     <Card className="w-full max-w-lg mx-auto border-none shadow-none">
//       <CardHeader>
//         <CardTitle>Create a New Post</CardTitle>
//       </CardHeader>
//       <CardContent>
//         <form onSubmit={handleSubmit} className="space-y-6">
//           <Textarea
//             placeholder="What's on your mind?"
//             value={text}
//             onChange={(e) => setText(e.target.value)}
//             rows={4}
//             disabled={isSubmitting}
//           />
          
//           {/* ✅ 3. Add UI for selecting the post space */}
//           {availableSpaces.length > 1 && (
//             <div>
//               <Label>Choose a Space</Label>
//               <RadioGroup
//                 value={space}
//                 onValueChange={(value) => setSpace(value as Gender)}
//                 className="flex items-center gap-4 mt-2"
//                 disabled={isSubmitting}
//               >
//                 {availableSpaces.map((s) => (
//                   <div key={s} className="flex items-center space-x-2">
//                     <RadioGroupItem value={s} id={s} />
//                     <Label htmlFor={s} className="capitalize font-normal cursor-pointer">{s}</Label>
//                   </div>
//                 ))}
//               </RadioGroup>
//             </div>
//           )}

//           <div className="grid grid-cols-2 gap-4">
//             <div>
//               <Label htmlFor="topics">Topics (comma-separated)</Label>
//               <Input id="topics" placeholder="e.g. sports, news" value={topics} onChange={(e) => setTopics(e.target.value)} disabled={isSubmitting} />
//             </div>
//             <div>
//               <Label htmlFor="tags">Tags (comma-separated)</Label>
//               <Input id="tags" placeholder="e.g. fun, travel" value={tags} onChange={(e) => setTags(e.target.value)} disabled={isSubmitting} />
//             </div>
//           </div>
//           <div>
//             <Label htmlFor="media">Add Photo or Video</Label>
//             <Input
//               id="media"
//               type="file"
//               accept="image/*,video/*"
//               onChange={(e) => setFile(e.target.files?.[0] || null)}
//               disabled={isSubmitting}
//             />
//             {file && <p className="text-sm text-muted-foreground mt-2">Selected: {file.name}</p>}
//           </div>
//           <Button type="submit" className="w-full" disabled={isSubmitting}>
//             {isSubmitting ? (
//               <Loader2 className="mr-2 h-4 w-4 animate-spin" />
//             ) : (
//               <Send className="mr-2 h-4 w-4" />
//             )}
//             {isSubmitting ? 'Posting...' : 'Post'}
//           </Button>
//         </form>
//       </CardContent>
//     </Card>
//   );
// }



'use client';

import { useState } from 'react';
import { useSession } from 'next-auth/react';
import { toast } from 'sonner';
import { upload } from '@imagekit/next';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Loader2, Send } from 'lucide-react';

type Gender = "male" | "female" | "trans" | "other" | "common";

interface CreatePostFormProps {
  onPostCreated?: (post: any) => void;
}

export function CreatePostForm({ onPostCreated }: CreatePostFormProps) {
  const { data: session } = useSession();
  
  const [text, setText] = useState('');
  const [topics, setTopics] = useState('');
  const [tags, setTags] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [space, setSpace] = useState<Gender>('common');

  const userGender = session?.user?.gender as Gender;
  const availableSpaces = 
    userGender && userGender !== 'common' && userGender !== 'other'
      ? ['common', userGender]
      : ['common'];

  const resetForm = () => {
    setText('');
    setTopics('');
    setTags('');
    setFile(null);
    setSpace('common');
  };

  const handleUpload = async () => {
    if (!file) return { url: '', fileType: '' };
    try {
      const authRes = await fetch('/api/imagekit-auth');
      if (!authRes.ok) throw new Error('Failed to authenticate with ImageKit');
      const authParams = await authRes.json();
      const response = await upload({ file, fileName: file.name, ...authParams });
      return { url: response.url, fileType: file.type.startsWith('image/') ? 'photo' : 'video' };
    } catch (error) {
      toast.error('Media upload failed. Please try again.');
      throw error;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!session) { toast.error('You must be logged in to post.'); return; }
    if (!text && !file) { toast.error('Your post must include text or media.'); return; }

    setIsSubmitting(true);
    try {
      const mediaData = { photo: '', video: '' };
      if (file) {
        const { url, fileType } = await handleUpload();
        if (fileType === 'photo') mediaData.photo = url;
        else mediaData.video = url;
      }

      const body = {
        text,
        ...mediaData,
        topics: topics.split(',').map(t => t.trim()).filter(Boolean),
        tags: tags.split(',').map(t => t.trim()).filter(Boolean),
        space, // Ensure the 'space' state is included in the body
      };

      const res = await fetch('/api/posts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.message || 'Failed to create post.');
      } 

      const newPost = await res.json();
      toast.success(`Post created in '${space}' space!`);
      resetForm();
      onPostCreated?.(newPost);

    } catch (error: any) {
      toast.error(error.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card className="w-full max-w-lg mx-auto border-none shadow-none">
      <CardHeader>
        <CardTitle>Create a New Post</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <Textarea
            placeholder="What's on your mind?"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={4}
            disabled={isSubmitting}
          />
          
          {availableSpaces.length > 1 && (
            <div>
              <Label>Choose a Space</Label>
              <RadioGroup
                value={space}
                onValueChange={(value) => setSpace(value as Gender)}
                className="flex items-center gap-4 mt-2"
                disabled={isSubmitting}
              >
                {availableSpaces.map((s) => (
                  <div key={s} className="flex items-center space-x-2">
                    <RadioGroupItem value={s} id={s} />
                    <Label htmlFor={s} className="capitalize font-normal cursor-pointer">{s}</Label>
                  </div>
                ))}
              </RadioGroup>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="topics">Topics (comma-separated)</Label>
              <Input id="topics" placeholder="e.g. sports, news" value={topics} onChange={(e) => setTopics(e.target.value)} disabled={isSubmitting} />
            </div>
            <div>
              <Label htmlFor="tags">Tags (comma-separated)</Label>
              <Input id="tags" placeholder="e.g. fun, travel" value={tags} onChange={(e) => setTags(e.target.value)} disabled={isSubmitting} />
            </div>
          </div>
          <div>
            <Label htmlFor="media">Add Photo or Video</Label>
            <Input
              id="media"
              type="file"
              accept="image/*,video/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              disabled={isSubmitting}
            />
            {file && <p className="text-sm text-muted-foreground mt-2">Selected: {file.name}</p>}
          </div>
          <Button type="submit" className="w-full" disabled={isSubmitting}>
            {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Send className="mr-2 h-4 w-4" />}
            {isSubmitting ? 'Posting...' : 'Post'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}