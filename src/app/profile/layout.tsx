import Link from 'next/link';
import { getServerSession } from 'next-auth';
import { authOptions } from '../api/auth/[...nextauth]/route';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Home, User, MessageCircle, LogOut } from 'lucide-react';
import { UploadModal } from '@/components/posts/UploadModal'; // Adjust path if needed
import { SearchModal } from '@/components/modals/SearchModal'; // 1. Import the new SearchModal
import { cn } from '@/lib/utils';

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getServerSession(authOptions);
  const username = session?.user?.username;

  // 2. Remove 'Search' from the navItems array
  const navItems = [
    { href: '/explore', icon: Home, label: 'Feed' }, // Changed from /explore
    { href: username ? `/profile/${username}` : '/profile', icon: User, label: 'Profile' },
    { href: '/chat', icon: MessageCircle, label: 'Chat' },
  ];

  return (
    <div className="flex h-screen bg-muted/10">
      {/* Sidebar for Desktop */}
      <aside className="hidden md:flex w-64 flex-col justify-between border-r bg-card shadow-sm">
        <div className="p-6">
          <h2 className="text-2xl font-bold mb-8 text-primary">Switch</h2>
          <ScrollArea className="h-[calc(100vh-12rem)]">
            <nav className="flex flex-col gap-3">
              {navItems.map((item) => (
                <Link key={item.href} href={item.href} className={cn("flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors", "hover:bg-primary/10 hover:text-primary")}>
                  <item.icon className="w-5 h-5" />
                  {item.label}
                </Link>
              ))}
              {/* 3. Add the SearchModal trigger */}
              <div className={cn("flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium cursor-pointer transition-colors", "hover:bg-primary/10 hover:text-primary")}>
                <SearchModal variant="desktop" />
              </div>
              <div className={cn("flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium cursor-pointer transition-colors", "hover:bg-primary/10 hover:text-primary")}>
                <UploadModal variant="desktop" />
              </div>
            </nav>
          </ScrollArea>
        </div>
        <div className="p-4">
          <form action="/api/auth/signout" method="POST">
            <Button variant="ghost" className="w-full justify-start gap-3" type="submit">
              <LogOut className="w-5 h-5" />
              Logout
            </Button>
          </form>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <div className="p-4 md:p-8">{children}</div>
      </main>

      {/* Mobile Bottom Navigation */}
      <nav className="md:hidden fixed bottom-4 left-1/2 -translate-x-1/2 w-[95%] max-w-md bg-card border rounded-2xl shadow-lg flex justify-around py-2 z-50 backdrop-blur-md">
        {navItems.map((item) => (
          <Link key={item.href} href={item.href} className="flex flex-col items-center text-muted-foreground hover:text-primary w-1/5">
            <item.icon className="w-5 h-5 mb-0.5" />
            <span className="text-[11px]">{item.label}</span>
          </Link>
        ))}
        {/* 4. Add the SearchModal trigger for mobile */}
        <div className="w-1/5"><SearchModal variant="mobile" /></div>
        <div className="w-1/5"><UploadModal variant="mobile" /></div>
        <form action="/api/auth/signout" method="POST" className="w-1/5">
          <button type="submit" className="flex flex-col items-center text-muted-foreground hover:text-primary w-full">
            <LogOut className="w-5 h-5 mb-0.5" />
            <span className="text-[11px]">Logout</span>
          </button>
        </form>
      </nav>
    </div>
  );
}