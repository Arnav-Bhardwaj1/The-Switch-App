'use client';

import { createContext, useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { io, Socket } from 'socket.io-client';

interface SocketContextType {
  socket: Socket | null;
  onlineUsers: string[];
}
const SocketContext = createContext<SocketContextType | undefined>(undefined);
export const useSocket = () => { /* ... same as before ... */ };

export const SocketProvider = ({ children }: { children: React.ReactNode }) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [onlineUsers, setOnlineUsers] = useState<string[]>([]);
  const { data: session } = useSession();

  useEffect(() => {
    if (session?.user?.id) {
      // ✅ Connect directly. No need for the fetch call.
      const newSocket = io(); // Connects to the same host and port

      newSocket.on('connect', () => {
        console.log('Socket connected:', newSocket.id);
        newSocket.emit('addUser', session.user.id);
      });

      newSocket.on('getOnlineUsers', (users: string[]) => {
        setOnlineUsers(users);
      });
      setSocket(newSocket);

      return () => {
        newSocket.disconnect();
      };
    } else {
      if (socket) {
        socket.disconnect();
        setSocket(null);
      }
    }
  }, [session, socket]);

  return (
    <SocketContext.Provider value={{ socket, onlineUsers }}>
      {children}
    </SocketContext.Provider>
  );
};