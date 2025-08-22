// import { ConversationList } from "@/components/chat/ConversationList";
// import { MessageView } from "@/components/chat/MessageView";
// import { SocketProvider } from "../context/SocketContext";

// export default function ChatPage() {
//   return (
//     // The SocketProvider wraps the entire chat interface
//     <SocketProvider>
//       <div className="flex h-[calc(100vh-8rem)]">
//         {/* Left Column: List of conversations */}
//         <div className="w-1/3 border-r">
//           <ConversationList />
//         </div>

//         {/* Right Column: The selected conversation's messages */}
//         <div className="flex-1">
//           <MessageView />
//         </div>
//       </div>
//     </SocketProvider>
//   );
// }

'use client';

import { useState, useEffect, useRef } from "react";
import { useSession } from "next-auth/react";
import { SocketProvider, useSocket } from "../context/SocketContext";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Loader2, Send, MessageCircle } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

// --- TYPE DEFINITIONS ---
interface Author { _id: string; name: string; username: string; profileImage?: string; }
interface Message { _id: string; conversationId: string; sender: Author; text: string; createdAt: string; }
type Conversation = { _id: string; participants: Author[]; messages: [{ text: string; }]; updatedAt: string; };

// --- CHAT CLIENT: The main component that holds all state and logic ---
function ChatClient() {
  const { data: session } = useSession();
  const { socket, onlineUsers } = useSocket();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState("");
  const [isLoadingConvos, setIsLoadingConvos] = useState(true);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const messageEndRef = useRef<HTMLDivElement>(null);

  // Fetch initial conversations
  useEffect(() => {
    const fetchConversations = async () => {
      setIsLoadingConvos(true);
      try {
        const res = await fetch('/api/conversations');
        if (!res.ok) throw new Error("Failed to fetch conversations");
        const data = await res.json();
        setConversations(data);
      } catch (error) {
        toast.error("Could not load your conversations.");
      } finally {
        setIsLoadingConvos(false);
      }
    };
    fetchConversations();
  }, []);

  // Fetch messages when a new conversation is selected
  useEffect(() => {
    if (!selectedConversation) return;
    const fetchMessages = async () => {
      setIsLoadingMessages(true);
      try {
        const res = await fetch(`/api/messages/${selectedConversation._id}`);
        if (!res.ok) throw new Error("Failed to fetch messages");
        const data = await res.json();
        setMessages(data);
      } catch (error) { toast.error("Could not load messages."); }
      finally { setIsLoadingMessages(false); }
    };
    fetchMessages();
  }, [selectedConversation]);

  // Listen for new messages in real-time
  useEffect(() => {
    if (!socket) return;
    const handleNewMessage = (message: Message) => {
      // If the message is for the currently open conversation, add it to the view
      if (message.conversationId === selectedConversation?._id) {
        setMessages(prev => [...prev, message]);
      }
      // Update the conversation list to show the new message and bring it to the top
      setConversations(prevConvos => {
        const convoToUpdate = prevConvos.find(c => c._id === message.conversationId);
        if (convoToUpdate) {
          const updatedConvo = { ...convoToUpdate, messages: [{ text: message.text }], updatedAt: message.createdAt };
          const otherConvos = prevConvos.filter(c => c._id !== message.conversationId);
          return [updatedConvo, ...otherConvos];
        }
        return prevConvos;
      });
    };
    socket.on("newMessage", handleNewMessage);
    return () => { socket.off("newMessage", handleNewMessage); };
  }, [socket, selectedConversation]);

  // Auto-scroll to the bottom of messages
  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMessage.trim() || !socket || !selectedConversation || !session) return;
    const otherUser = selectedConversation.participants.find(p => p._id !== session.user.id);
    if (!otherUser) return;

    socket.emit("sendMessage", {
      conversationId: selectedConversation._id,
      senderId: session.user.id,
      receiverId: otherUser._id,
      text: newMessage,
    });
    setNewMessage("");
  };

  return (
    <div className="flex h-[calc(100vh-8rem)]">
      {/* Left Column: Conversation List */}
      <div className="w-1/3 border-r flex flex-col">
        <h2 className="text-xl font-bold p-4 border-b">Messages</h2>
        {isLoadingConvos ? (
          <div className="flex justify-center items-center h-full"><Loader2 className="animate-spin" /></div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            {conversations.map((convo) => {
              const otherUser = convo.participants.find(p => p._id !== session?.user?.id);
              if (!otherUser) return null;
              const isOnline = onlineUsers.includes(otherUser._id);
              return (
                <div key={convo._id} onClick={() => setSelectedConversation(convo)} className={cn("flex items-center p-4 gap-3 cursor-pointer transition-colors", selectedConversation?._id === convo._id ? "bg-muted" : "hover:bg-muted/50")}>
                  <div className="relative">
                    <Avatar><AvatarImage src={otherUser.profileImage} /><AvatarFallback>{otherUser.name[0]}</AvatarFallback></Avatar>
                    {isOnline && <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white" />}
                  </div>
                  <div className="flex-1 overflow-hidden">
                    <p className="font-semibold truncate">{otherUser.name}</p>
                    <p className="text-sm text-muted-foreground truncate">{convo.messages[0]?.text || "Start a conversation"}</p>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Right Column: Message View */}
      <div className="flex-1 flex flex-col">
        {!selectedConversation ? (
          <div className="flex flex-col h-full items-center justify-center bg-muted/50">
            <MessageCircle className="w-16 h-16 text-muted-foreground/50" />
            <h2 className="text-xl font-semibold mt-4 text-muted-foreground">Select a conversation</h2>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-3 p-3 border-b">
              <Avatar><AvatarImage src={selectedConversation.participants.find(p => p._id !== session?.user?.id)?.profileImage} /><AvatarFallback>{selectedConversation.participants.find(p => p._id !== session?.user?.id)?.name[0]}</AvatarFallback></Avatar>
              <h2 className="font-semibold">{selectedConversation.participants.find(p => p._id !== session?.user?.id)?.name}</h2>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {isLoadingMessages ? <div className="flex justify-center"><Loader2 className="animate-spin" /></div> :
                messages.map((msg) => (
                  <div key={msg._id} className={`flex ${msg.sender._id === session?.user?.id ? 'justify-end' : 'justify-start'}`}>
                    <div className={`p-3 rounded-lg max-w-md ${msg.sender._id === session?.user?.id ? 'bg-primary text-primary-foreground' : 'bg-muted'}`}>
                      {msg.text}
                    </div>
                  </div>
                ))
              }
              <div ref={messageEndRef} />
            </div>
            <div className="p-4 border-t">
              <form onSubmit={handleSendMessage} className="flex gap-2">
                <Input value={newMessage} onChange={(e) => setNewMessage(e.target.value)} placeholder="Type a message..." />
                <Button type="submit" disabled={!newMessage.trim()}><Send className="w-4 h-4" /></Button>
              </form>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// --- PAGE WRAPPER: This ensures ChatClient is always inside the provider ---
export default function ChatPage() {
  return (
    <SocketProvider>
      <ChatClient />
    </SocketProvider>
  );
}