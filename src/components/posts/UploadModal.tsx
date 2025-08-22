'use client';

import { useState } from 'react';
import { CreatePostForm } from '@/components/posts/CreatePostForm';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { PlusSquare } from 'lucide-react'; // Assuming LogOut is needed elsewhere

// We define different visual styles for the trigger
interface UploadModalProps {
  variant: 'desktop' | 'mobile';
}

export function UploadModal({ variant }: UploadModalProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handlePostCreated = () => {
    // This function is passed to the form to close the modal on success
    setIsOpen(false);
    // You could also add logic here to refresh the user's feed
  };

  // The DialogTrigger is the element that the user clicks.
  // We style it differently based on the 'variant' prop.
  const trigger =
    variant === 'desktop' ? (
      <button className="flex items-center gap-2 text-gray-800 hover:text-black w-full text-left">
        <PlusSquare className="w-5 h-5" />
        Upload
      </button>
    ) : (
      <button className="flex flex-col items-center text-gray-700 hover:text-black text-sm">
        <PlusSquare className="w-5 h-5 mb-0.5" />
        <span className="text-xs">Upload</span>
      </button>
    );

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-[425px] md:max-w-lg">
        <DialogHeader>
          <DialogTitle>Create a New Post</DialogTitle>
        </DialogHeader>
        {/* We place the form inside the modal content */}
        <div className="py-4">
          <CreatePostForm onPostCreated={handlePostCreated} />
        </div>
      </DialogContent>
    </Dialog>
  );
}