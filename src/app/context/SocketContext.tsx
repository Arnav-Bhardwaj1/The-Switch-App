// 'use client';

// import { createContext, useContext, useEffect, useState } from 'react';
// import { useSession } from 'next-auth/react';
// import { io, Socket } from 'socket.io-client';

// // Define the shape of the context
// interface SocketContextType {
//   socket: Socket | null;
//   onlineUsers: string[];
// }

// const SocketContext = createContext<SocketContextType | undefined>(undefined);

// // Custom hook for easy access to the context
// export const useSocket = () => {
//   const context = useContext(SocketContext);
//   if (context === undefined) {
//     throw new Error('useSocket must be used within a SocketProvider');
//   }
//   return context;
// };

// export const SocketProvider = ({ children }: { children: React.ReactNode }) => {
//   const [socket, setSocket] = useState<Socket | null>(null);
//   const [onlineUsers, setOnlineUsers] = useState<string[]>([]);
//   const { data: session } = useSession();

//   useEffect(() => {
//     // Only establish connection if the user is authenticated
//     if (session?.user?.id) {
//       // This is a common pattern to "wake up" the socket server on serverless envs
//       const initializeSocket = async () => {
//         await fetch('/api/socket/io');

//         const newSocket = io({
//           path: '/api/socket/io',
//           // You might need to add auth details here later for security
//         });

//         newSocket.on('connect', () => {
//           console.log('Socket connected:', newSocket.id);
//           // Tell the server who this is
//           newSocket.emit('addUser', session.user.id);
//         });

//         // Listen for the list of online users
//         newSocket.on('getOnlineUsers', (users: string[]) => {
//           setOnlineUsers(users);
//         });

//         setSocket(newSocket);
//       };

//       initializeSocket();

//       // Cleanup on component unmount
//       return () => {
//         socket?.disconnect();
//         setSocket(null);
//       };
//     } else {
//       // If user logs out, disconnect the socket
//       if (socket) {
//         socket.disconnect();
//         setSocket(null);
//       }
//     }
//   // eslint-disable-next-line react-hooks/exhaustive-deps
//   }, [session]);

//   return (
//     <SocketContext.Provider value={{ socket, onlineUsers }}>
//       {children}
//     </SocketContext.Provider>
//   );
// };


'use client';

import { createContext, useContext, useEffect, useState } from 'react';
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
  }, [session]);

  return (
    <SocketContext.Provider value={{ socket, onlineUsers }}>
      {children}
    </SocketContext.Provider>
  );
};