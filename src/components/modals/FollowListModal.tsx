'use client';

import Link from 'next/link'; // 1. Import the Link component
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { UserPlus, UserCheck } from "lucide-react";

type ListUser = {
  _id: string;
  name: string;
  username: string;
  profileImage?: string;
};

interface FollowListModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: 'Followers' | 'Following';
  users: ListUser[];
  currentUserId?: string;
  sessionUserFollowing: string[];
  onFollowToggle: (targetUserId: string) => void;
}

export function FollowListModal({
  isOpen,
  onClose,
  title,
  users,
  currentUserId,
  sessionUserFollowing,
  onFollowToggle,
}: FollowListModalProps) {
  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        <ScrollArea className="h-72 w-full pr-4">
          <div className="space-y-4">
            {users.length > 0 ? (
              users.map((user) => {
                const isFollowing = sessionUserFollowing.includes(user._id);
                const isCurrentUser = user._id === currentUserId;

                return (
                  <div key={user._id} className="flex items-center justify-between">
                    {/* 2. Wrap the user info in a Link component */}
                    <Link 
                      href={`/profile/${user.username}`} 
                      className="flex items-center gap-3"
                      onClick={onClose} // Close modal on click
                    >
                      <Avatar>
                        <AvatarImage src={user.profileImage} />
                        <AvatarFallback>{user.name?.[0]}</AvatarFallback>
                      </Avatar>
                      <div>
                        <p className="font-semibold text-sm hover:underline">{user.name}</p>
                        {/* 3. The username will now be visible because the API is sending it */}
                        <p className="text-xs text-muted-foreground">@{user.username}</p>
                      </div>
                    </Link>

                    {/* The button remains outside the link */}
                    {!isCurrentUser && (
                       <Button
                        variant={isFollowing ? 'secondary' : 'default'}
                        size="sm"
                        onClick={() => onFollowToggle(user._id)}
                      >
                        {isFollowing ? <UserCheck className="mr-2 h-4 w-4" /> : <UserPlus className="mr-2 h-4 w-4" />}
                        {isFollowing ? 'Following' : 'Follow'}
                      </Button>
                    )}
                  </div>
                );
              })
            ) : (
              <p className="text-center text-muted-foreground py-8">No users to display.</p>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}