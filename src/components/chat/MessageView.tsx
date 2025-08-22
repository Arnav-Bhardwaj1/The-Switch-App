// 'use client';

// import { MessageCircle } from "lucide-react";

// export function MessageView() {
//     // For now, this is a placeholder. In the next step, we'll make it
//     // display messages from a selected conversation.
//   return (
//     <div className="flex flex-col h-full items-center justify-center bg-muted/50">
//         <MessageCircle className="w-16 h-16 text-muted-foreground/50" />
//         <h2 className="text-xl font-semibold mt-4 text-muted-foreground">Select a conversation</h2>
//         <p className="text-muted-foreground">Choose one of your existing conversations to start chatting.</p>
//     </div>
//   );
// }


'use client';

import { useEffect, useState, useRef } from "react";
import { useSession } from "next-auth/react";
import { Conversation } from "@/app/chat/page";
import { useSocket } from "@/app/context/SocketContext";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Loader2, Send, MessageCircle } from "lucide-react";
import { toast } from "sonner";

interface Message {
  _id: string;
  sender: { _id: string; name: string; profileImage?: string; };
  text: string;
  createdAt: string;
}

interface MessageViewProps {
  selectedConversation: Conversation | null;
}

export function MessageView({ selectedConversation }: MessageViewProps) {
  const { data: session } = useSession();
  const { socket } = useSocket();
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messageEndRef = useRef<HTMLDivElement>(null);

  // Scroll to the bottom of the messages
  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Fetch message history when a conversation is selected
  useEffect(() => {
    if (!selectedConversation) return;
    const fetchMessages = async () => {
      setIsLoading(true);
      try {
        const res = await fetch(`/api/messages/${selectedConversation._id}`);
        if (!res.ok) throw new Error("Failed to fetch messages");
        const data = await res.json();
        setMessages(data);
      } catch (error) {
        toast.error("Could not load messages.");
      } finally {
        setIsLoading(false);
      }
    };
    fetchMessages();
  }, [selectedConversation]);

  // Listen for new messages in real-time
  useEffect(() => {
    if (!socket) return;
    socket.on("newMessage", (message: Message) => {
      // Add the new message if it belongs to the current conversation
      if (message.conversationId === selectedConversation?._id) {
          setMessages((prev) => [...prev, message]);
      }
    });
    return () => { socket.off("newMessage"); };
  }, [socket, selectedConversation]);


  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMessage.trim() || !socket || !selectedConversation) return;

    const otherUser = selectedConversation.participants.find(p => p._id !== session?.user?.id);
    if (!otherUser) return;

    // The message object to be saved in the DB and sent via socket
    const messageData = {
        conversationId: selectedConversation._id,
        sender: session?.user?.id,
        text: newMessage,
    };

    // Send message via Socket.IO for real-time delivery
    socket.emit("sendMessage", {
        receiverId: otherUser._id,
        message: messageData
    });
    
    // Also send to an API to save it in the database
    // (A more advanced setup would have the socket server handle saving to DB)
    await fetch('/api/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(messageData),
    });

    setMessages((prev) => [...prev, messageData as any]);
    setNewMessage("");
  };

  if (!selectedConversation) {
    return (
      <div className="flex flex-col h-full items-center justify-center bg-muted/50">
        <MessageCircle className="w-16 h-16 text-muted-foreground/50" />
        <h2 className="text-xl font-semibold mt-4 text-muted-foreground">Select a conversation</h2>
        <p className="text-muted-foreground">Choose one of your existing conversations to start chatting.</p>
      </div>
    );
  }
  
  const otherUser = selectedConversation.participants.find(p => p._id !== session?.user?.id)!;

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-3 p-3 border-b">
        <Avatar><AvatarImage src={otherUser.profileImage}/><AvatarFallback>{otherUser.name[0]}</AvatarFallback></Avatar>
        <h2 className="font-semibold">{otherUser.name}</h2>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {isLoading && <div className="flex justify-center"><Loader2 className="animate-spin"/></div>}
        {messages.map((msg) => (
          <div key={msg._id} className={`flex ${msg.sender._id === session?.user?.id ? 'justify-end' : 'justify-start'}`}>
            <div className={`p-3 rounded-lg max-w-md ${msg.sender._id === session?.user?.id ? 'bg-primary text-primary-foreground' : 'bg-muted'}`}>
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={messageEndRef} />
      </div>
      <div className="p-4 border-t">
        <form onSubmit={handleSendMessage} className="flex gap-2">
          <Input value={newMessage} onChange={(e) => setNewMessage(e.target.value)} placeholder="Type a message..." />
          <Button type="submit"><Send className="w-4 h-4" /></Button>
        </form>
      </div>
    </div>
  );
}