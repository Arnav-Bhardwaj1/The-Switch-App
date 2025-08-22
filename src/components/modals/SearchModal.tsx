'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Loader2, Search as SearchIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

type UserSearchResult = {
  _id: string;
  name: string;
  username: string;
  profileImage?: string;
};

// We define different visual styles for the trigger button
interface SearchModalProps {
  variant: 'desktop' | 'mobile';
}

export function SearchModal({ variant }: SearchModalProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<UserSearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  // Debounced search function
  const searchUsers = useCallback(async (searchQuery: string) => {
    if (searchQuery.length < 2) {
      setResults([]);
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    setHasSearched(true);
    try {
      const res = await fetch(`/api/users/search/${searchQuery}`);
      if (!res.ok) throw new Error('Search failed');
      const data = await res.json();
      setResults(data);
    } catch (error) {
      console.error(error);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    const debounceTimer = setTimeout(() => {
      if (query) {
        searchUsers(query);
      } else {
        setResults([]);
        setHasSearched(false);
      }
    }, 500);

    return () => clearTimeout(debounceTimer);
  }, [query, searchUsers]);
  
  // Reset state when modal is closed
  useEffect(() => {
    if (!isOpen) {
        setQuery('');
        setResults([]);
        setHasSearched(false);
    }
  }, [isOpen]);

  const trigger =
    variant === 'desktop' ? (
      <div className="flex items-center gap-3 w-full">
        <SearchIcon className="w-5 h-5" />
        Search
      </div>
    ) : (
      <div className="flex flex-col items-center text-muted-foreground hover:text-primary w-full">
        <SearchIcon className="w-5 h-5 mb-0.5" />
        <span className="text-[11px]">Search</span>
      </div>
    );

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Search for Users</DialogTitle>
        </DialogHeader>
        <div className="relative">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search by name or username..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-9"
          />
        </div>
        <div className="space-y-2 max-h-80 overflow-y-auto pr-2">
          {isLoading && <div className="flex justify-center py-4"><Loader2 className="w-6 h-6 animate-spin text-primary" /></div>}
          {!isLoading && hasSearched && results.length === 0 && <p className="text-center text-sm text-muted-foreground py-4">No users found.</p>}
          {!isLoading && results.length > 0 && results.map((user) => (
            <Link href={`/profile/${user.username}`} key={user._id} onClick={() => setIsOpen(false)}>
              <div className="p-2 flex items-center gap-3 rounded-md hover:bg-muted/50 transition-colors">
                <Avatar className="h-10 w-10">
                  <AvatarImage src={user.profileImage} alt={user.name} />
                  <AvatarFallback>{user.name[0]}</AvatarFallback>
                </Avatar>
                <div>
                  <p className="font-semibold text-sm">{user.name}</p>
                  <p className="text-xs text-muted-foreground">@{user.username}</p>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}