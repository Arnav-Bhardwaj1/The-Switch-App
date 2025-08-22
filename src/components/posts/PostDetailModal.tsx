'use client';

import { useState, useEffect, useRef } from 'react';
import { useSession } from 'next-auth/react';
import Link from 'next/link';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { toast } from 'sonner';
import { Heart, MessageCircle, Share2, Bookmark, ChevronLeft, ChevronRight, X, Loader2 } from 'lucide-react';

// --- TYPE INTERFACES ---

interface Author {
  _id: string;
  name: string;
  username: string;
  profileImage?: string;
}

interface Comment {
  _id: string;
  author: Author;
  text: string;
  createdAt: string;
}

interface DetailedPost {
  _id: string;
  author?: Author;
  text: string;
  photo?: string;
  video?: string;
  likes: string[];
  savedBy: string[];
  comments: Comment[];
}

interface PostDetailModalProps {
  posts: DetailedPost[];
  startIndex: number;
  isOpen: boolean;
  onClose: () => void;
  onPostUpdate: (updatedPost: any) => void;
  readOnly?: boolean;
}

// --- MAIN MODAL COMPONENT ---

export function PostDetailModal({ posts, startIndex, isOpen, onClose, onPostUpdate, readOnly = false }: PostDetailModalProps) {
  const { data: session } = useSession();
  const commentInputRef = useRef<HTMLInputElement>(null);
  const commentListRef = useRef<HTMLDivElement>(null);

  const [currentIndex, setCurrentIndex] = useState(startIndex);
  const [newComment, setNewComment] = useState("");
  const [isCommenting, setIsCommenting] = useState(false);
  
  const currentPost = posts[currentIndex]; 
  const hasMedia = !!(currentPost?.photo || currentPost?.video);

  useEffect(() => {
    // Reset the index when the modal is opened with a new start index
    setCurrentIndex(startIndex);
  }, [startIndex, isOpen]);

  useEffect(() => {
    // Auto-scroll to the bottom of the comments when a new one is added
    if (commentListRef.current) {
      commentListRef.current.scrollTop = commentListRef.current.scrollHeight;
    }
  }, [currentPost?.comments]);

  if (!isOpen || !currentPost) return null;

  const currentUserId = (session?.user as any)?.id;
  const isLiked = currentUserId ? currentPost.likes.includes(currentUserId) : false;
  const isSaved = currentUserId ? currentPost.savedBy?.includes(currentUserId) : false;

  const optimisticUpdate = (action: 'like' | 'save' | 'comment', payload?: any) => {
    const updatedPost = { ...currentPost };
    switch (action) {
      case 'like':
        updatedPost.likes = isLiked
          ? currentPost.likes.filter(id => id !== currentUserId)
          : [...currentPost.likes, currentUserId!];
        break;
      case 'save':
        updatedPost.savedBy = isSaved
          ? currentPost.savedBy.filter(id => id !== currentUserId)
          : [...(currentPost.savedBy || []), currentUserId!];
        break;
      case 'comment':
        // This defensive check prevents the app from crashing
        updatedPost.comments = [...(currentPost.comments || []), payload];
        break;
    }
    onPostUpdate(updatedPost);
  };

  const handleLike = async () => {
    if (!session) { toast.error("Please log in to like posts."); return; }
    if (readOnly) { return; }
    optimisticUpdate('like');
    try {
      const res = await fetch(`/api/posts/${currentPost._id}/like`, { method: 'POST' });
       if (!res.ok) throw new Error("Server failed to process like.");
    } catch (error) {
      toast.error("Failed to update like.");
      optimisticUpdate('like'); // Revert on error
    }
  };
  
  const handleSave = async () => {
    if (!session) { toast.error("Please log in to save posts."); return; }
    if (readOnly) { return; }
    optimisticUpdate('save');
    try {
      const res = await fetch(`/api/posts/${currentPost._id}/save`, { method: 'POST' });
      if (!res.ok) throw new Error("Server failed to process save.");
    } catch (error) {
      toast.error("Failed to update save.");
      optimisticUpdate('save'); // Revert on error
    }
  };
  // ✅ THIS FUNCTION IS UPDATED WITH LOGS
  const handleComment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!session) { toast.error("Please log in to comment."); return; }
    if (readOnly) { return; }
    if (!newComment.trim()) return;

    setIsCommenting(true);
    console.log("--- Starting comment submission ---");

    try {
      console.log(`Step 1: Sending fetch request to /api/posts/${currentPost._id}/comment`);
      const res = await fetch(`/api/posts/${currentPost._id}/comment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: newComment }),
      });
      console.log("Step 2: Received response from API. Status:", res.status, "OK:", res.ok);

      if (!res.ok) {
        const errorBody = await res.text();
        console.error("API returned an error response body:", errorBody);
        throw new Error(`Server returned status ${res.status}`);
      }

      console.log("Step 3: Trying to parse response as JSON...");
      const newlyCreatedComment = await res.json();
      console.log("Step 4: JSON parsed successfully:", newlyCreatedComment);

      console.log("Step 5: Calling optimisticUpdate with the new comment...");
      optimisticUpdate('comment', newlyCreatedComment);
      console.log("Step 6: optimisticUpdate finished successfully.");

      setNewComment("");
    } catch (error) {
      // ❗ THIS WILL SHOW US THE REAL JAVASCRIPT ERROR
      console.error("--- CRITICAL ERROR IN handleComment ---", error);
      toast.error("An error occurred. See browser console for details.");
    } finally {
      setIsCommenting(false);
      console.log("--- Finished comment submission ---");
    }
  };


  const goToNext = () => setCurrentIndex(i => Math.min(i + 1, posts.length - 1));
  const goToPrev = () => setCurrentIndex(i => Math.max(i - 1, 0));

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className={`w-[95vw] max-w-4xl h-[92vh] p-0 ${hasMedia ? 'grid grid-cols-1 md:grid-cols-2' : ''} gap-0`}>
        <Button variant="ghost" size="icon" className="absolute top-2 right-2 z-20 text-white md:text-black bg-black/20 md:bg-transparent hover:bg-black/40" onClick={onClose}><X /></Button>
        {currentIndex > 0 && <Button variant="ghost" size="icon" className="absolute left-2 top-1/2 -translate-y-1/2 z-20 text-white bg-black/20 hover:bg-black/40" onClick={goToPrev}><ChevronLeft /></Button>}
        {currentIndex < posts.length - 1 && <Button variant="ghost" size="icon" className="absolute right-2 md:right-[50%] md:mr-2 top-1/2 -translate-y-1/2 z-20 text-white bg-black/20 hover:bg-black/40" onClick={goToNext}><ChevronRight /></Button>}
        
        {hasMedia && (
          <div className="bg-black flex items-center justify-center overflow-hidden md:rounded-l-lg">
            {currentPost.photo && <img src={currentPost.photo} alt="Post" className="object-contain max-h-full max-w-full" />}
            {currentPost.video && <video src={currentPost.video} controls className="object-contain max-h-full max-w-full" />}
          </div>
        )}
        
        <div className={`flex flex-col p-4 bg-white ${hasMedia ? 'md:rounded-r-lg' : 'rounded-lg'}`}>
          <div className="flex items-center gap-3 border-b pb-4">
            <Avatar><AvatarImage src={readOnly ? undefined : currentPost.author?.profileImage} /><AvatarFallback>{readOnly ? 'A' : (currentPost.author?.name ? currentPost.author.name[0].toUpperCase() : 'U')}</AvatarFallback></Avatar>
            <div>
              {readOnly ? (
                <p className="font-semibold">Anonymous</p>
              ) : (
                <Link href={currentPost.author?.username ? `/profile/${currentPost.author.username}` : '#'} className="font-semibold hover:underline">
                  {currentPost.author?.username || 'Unknown User'}
                </Link>
              )}
            </div>
          </div>
          
          <div ref={commentListRef} className="flex-1 py-4 overflow-y-auto space-y-4">
            {currentPost.text && (
              <div className="text-sm">
                <p>{currentPost.text}</p>
              </div>
            )}

            {(currentPost.comments?.length ?? 0) > 0 && (
              <div className="pt-4 space-y-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-500">Comments</p>
                {currentPost.comments?.map(comment => (
                  comment && comment.author && (
                    <div key={comment._id} className="flex items-start gap-3 text-sm">
                      <Avatar className="w-8 h-8">
                        <AvatarImage src={readOnly ? undefined : comment.author.profileImage} />
                        <AvatarFallback>{readOnly ? 'A' : (comment.author.name ? comment.author.name[0].toUpperCase() : '?')}</AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <p>
                          {readOnly ? (
                            <span className="font-semibold mr-2">Anonymous</span>
                          ) : (
                            <Link href={comment.author.username ? `/profile/${comment.author.username}` : '#'} className="font-semibold mr-2 hover:underline">
                              {comment.author.username}
                            </Link>
                          )}
                          {comment.text}
                        </p>
                      </div>
                    </div>
                  )
                ))}
              </div>
            )}
          </div>

          <div className="border-t pt-2 mt-auto">
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="icon" onClick={handleLike} disabled={!session || readOnly}><Heart className={isLiked ? 'text-red-500 fill-current' : ''} /></Button>
              <Button variant="ghost" size="icon" onClick={() => commentInputRef.current?.focus()} disabled={readOnly}><MessageCircle /></Button>
              <Button variant="ghost" size="icon" disabled={readOnly}><Share2 /></Button>
              <div className="flex-grow" />
              <Button variant="ghost" size="icon" onClick={handleSave} disabled={!session || readOnly}><Bookmark className={isSaved ? 'text-blue-500 fill-current' : ''} /></Button>
            </div>
            <p className="text-sm font-semibold px-2 mt-1">{currentPost.likes.length} likes</p>
            <form onSubmit={handleComment} className="flex gap-2 mt-2 px-2">
               <Input ref={commentInputRef} placeholder="Add a comment..." value={newComment} onChange={(e) => setNewComment(e.target.value)} disabled={isCommenting || readOnly}/>
               <Button type="submit" disabled={readOnly || isCommenting || !newComment.trim()}>
                 {isCommenting ? <Loader2 className="w-4 h-4 animate-spin"/> : "Post"}
               </Button>
            </form>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}