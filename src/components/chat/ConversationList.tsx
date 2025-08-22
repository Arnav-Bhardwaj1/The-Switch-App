// 'use client';

// import { useEffect, useState } from "react";
// import { useSession } from "next-auth/react";
// import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
// import { Loader2 } from "lucide-react";
// import { toast } from "sonner";

// // Define a type for the conversations we expect from the API
// type Conversation = {
//   _id: string;
//   participants: [{
//     _id: string;
//     name: string;
//     username: string;
//     profileImage?: string;
//   }];
//   messages: [{
//     text: string;
//   }];
//   updatedAt: string;
// };

// export function ConversationList() {
//   const { data: session } = useSession();
//   const [conversations, setConversations] = useState<Conversation[]>([]);
//   const [isLoading, setIsLoading] = useState(true);

//   useEffect(() => {
//     const fetchConversations = async () => {
//       try {
//         const res = await fetch('/api/conversations');
//         if (!res.ok) throw new Error("Failed to fetch conversations");
//         const data = await res.json();
//         setConversations(data);
//       } catch (error) {
//         toast.error("Could not load your conversations.");
//       } finally {
//         setIsLoading(false);
//       }
//     };
//     fetchConversations();
//   }, []);

//   if (isLoading) {
//     return <div className="flex justify-center items-center h-full"><Loader2 className="animate-spin" /></div>;
//   }

//   return (
//     <div className="flex flex-col h-full">
//       <h2 className="text-xl font-bold p-4 border-b">Messages</h2>
//       <div className="flex-1 overflow-y-auto">
//         {conversations.map((convo) => {
//           // Find the other participant in the conversation
//           const otherUser = convo.participants.find(p => p._id !== session?.user?.id);
//           if (!otherUser) return null;

//           return (
//             <div key={convo._id} className="flex items-center p-4 gap-3 hover:bg-muted cursor-pointer transition-colors">
//               <Avatar>
//                 <AvatarImage src={otherUser.profileImage} />
//                 <AvatarFallback>{otherUser.name[0]}</AvatarFallback>
//               </Avatar>
//               <div className="flex-1 overflow-hidden">
//                 <p className="font-semibold truncate">{otherUser.name}</p>
//                 <p className="text-sm text-muted-foreground truncate">
//                   {convo.messages[0]?.text || "No messages yet"}
//                 </p>
//               </div>
//             </div>
//           );
//         })}
//       </div>
//     </div>
//   );
// }

'use client';

import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useSocket } from "@/app/context/SocketContext";
import { Conversation } from "@/app/chat/page";
import { cn } from "@/lib/utils";

interface ConversationListProps {
  onSelectConversation: (conversation: Conversation) => void;
  selectedConversation: Conversation | null; // Add this prop to know which chat is active
}

export function ConversationList({ onSelectConversation, selectedConversation }: ConversationListProps) {
  const { data: session } = useSession();
  const { socket, onlineUsers } = useSocket(); // Get socket and online users
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch initial list of conversations
  useEffect(() => {
    const fetchConversations = async () => {
      setIsLoading(true);
      try {
        const res = await fetch('/api/conversations');
        if (!res.ok) throw new Error("Failed to fetch conversations");
        const data = await res.json();
        setConversations(data);
      } catch (error) {
        toast.error("Could not load your conversations.");
      } finally {
        setIsLoading(false);
      }
    };
    fetchConversations();
  }, []);

  // Listen for real-time updates to conversations
  useEffect(() => {
    if (!socket) return;

    const handleNewMessage = (newMessage: any) => {
      setConversations(prevConvos => {
        // Find the conversation that this message belongs to
        const convoToUpdate = prevConvos.find(c => c._id === newMessage.conversationId);
        if (convoToUpdate) {
          // Update the last message preview
          const updatedConvo = {
            ...convoToUpdate,
            messages: [{ text: newMessage.text }],
            updatedAt: newMessage.createdAt, // Use the new message's timestamp
          };
          // Remove the old conversation and put the updated one at the top
          const otherConvos = prevConvos.filter(c => c._id !== newMessage.conversationId);
          return [updatedConvo, ...otherConvos];
        }
        return prevConvos;
      });
    };

    socket.on("newMessage", handleNewMessage);

    return () => {
      socket.off("newMessage", handleNewMessage);
    };
  }, [socket]);


  if (isLoading) {
    return <div className="flex justify-center items-center h-full"><Loader2 className="animate-spin" /></div>;
  }

  return (
    <div className="flex flex-col h-full">
      <h2 className="text-xl font-bold p-4 border-b">Messages</h2>
      <div className="flex-1 overflow-y-auto">
        {conversations.length === 0 && (
            <p className="text-center text-muted-foreground p-8">No conversations yet.</p>
        )}
        {conversations.map((convo) => {
          const otherUser = convo.participants.find(p => p._id !== session?.user?.id);
          if (!otherUser) return null;

          const isOnline = onlineUsers.includes(otherUser._id);

          return (
            <div 
              key={convo._id} 
              className={cn(
                "flex items-center p-4 gap-3 cursor-pointer transition-colors",
                selectedConversation?._id === convo._id ? "bg-muted" : "hover:bg-muted/50"
              )}
              onClick={() => onSelectConversation(convo)}
            >
              <div className="relative">
                <Avatar>
                  <AvatarImage src={otherUser.profileImage} />
                  <AvatarFallback>{otherUser.name[0]}</AvatarFallback>
                </Avatar>
                {isOnline && (
                  <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white" />
                )}
              </div>
              <div className="flex-1 overflow-hidden">
                <p className="font-semibold truncate">{otherUser.name}</p>
                <p className="text-sm text-muted-foreground truncate">
                  {convo.messages[0]?.text || "Start a conversation"}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}