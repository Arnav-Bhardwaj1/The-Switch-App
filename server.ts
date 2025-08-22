import { createServer } from "http";
import { parse } from "url";
import next from "next";
import { Server } from "socket.io";

const dev = process.env.NODE_ENV !== "production";
const hostname = "localhost";
const port = 3000;

// when using middleware `hostname` and `port` must be provided below
const app = next({ dev, hostname, port });
const handle = app.getRequestHandler();

// A map to store userId and their corresponding socketId
const userSocketMap = new Map<string, string>();

app.prepare().then(() => {
  const httpServer = createServer(handle);

  const io = new Server(httpServer);

  io.on("connection", (socket) => {
    console.log("✅ User connected:", socket.id);

    // Get the userId from the client and map it to the socketId
    socket.on("addUser", (userId) => {
        if (userId) {
            userSocketMap.set(userId, socket.id);
            // Send the list of online users to all clients
            io.emit("getOnlineUsers", Array.from(userSocketMap.keys()));
            console.log(`User ${userId} added with socket ${socket.id}. Online users: ${userSocketMap.size}`);
        }
    });

    // Handle joining a conversation room
    socket.on("joinConversation", (conversationId) => {
        socket.join(conversationId);
        console.log(`Socket ${socket.id} joined room ${conversationId}`);
    });

    // Handle sending a message
    socket.on("sendMessage", (data) => {
        const { receiverId, message } = data;
        const receiverSocketId = userSocketMap.get(receiverId);
        if (receiverSocketId) {
            // Send the message to the specific receiver
            io.to(receiverSocketId).emit("newMessage", message);
        }
    });

    socket.on("disconnect", () => {
        // Find which user disconnected and remove them from the map
        for (const [userId, socketId] of userSocketMap.entries()) {
            if (socketId === socket.id) {
                userSocketMap.delete(userId);
                break;
            }
        }
        // Update the online users list for everyone
        io.emit("getOnlineUsers", Array.from(userSocketMap.keys()));
        console.log(`❌ User disconnected: ${socket.id}. Online users: ${userSocketMap.size}`);
    });
  });

  httpServer
    .once("error", (err) => {
      console.error(err);
      process.exit(1);
    })
    .listen(port, () => {
      console.log(`> Ready on http://${hostname}:${port}`);
    });
});


// import { createServer } from "http";
// import next from "next";
// import { Server } from "socket.io";
// // Import your Mongoose models
// import { Message } from "@/models/Message";
// import { Conversation } from "@/models/Conversation";
// import { User } from "@/models/User";
// import { connectToDB } from '@/lib/mongodb';

// const dev = process.env.NODE_ENV !== "production";
// const hostname = "localhost";
// const port = 3000;

// const app = next({ dev, hostname, port });
// const handle = app.getRequestHandler();

// const userSocketMap = new Map<string, string>();

// app.prepare().then(() => {
//   const httpServer = createServer(handle);
//   const io = new Server(httpServer);

//   io.on("connection", (socket) => {
//     // ... ('addUser', 'joinConversation', 'disconnect' events are the same)

//     // ✅ UPDATED 'sendMessage' HANDLER
//     socket.on("sendMessage", async (data) => {
//       const { conversationId, senderId, receiverId, text } = data;

//       try {
//         await connectToDB(); // Ensure DB is connected

//         // 1. Create and save the new message
//         const newMessage = new Message({
//           conversationId,
//           sender: senderId,
//           text,
//         });
//         const savedMessage = await newMessage.save();
        
//         // Populate sender details
//         const populatedMessage = await Message.findById(savedMessage._id).populate('sender', 'name username profileImage').lean();

//         // 2. Add message to the conversation
//         await Conversation.findByIdAndUpdate(conversationId, {
//           $push: { messages: savedMessage._id },
//         });

//         // 3. Send the message in real-time to the receiver
//         const receiverSocketId = userSocketMap.get(receiverId);
//         if (receiverSocketId) {
//           io.to(receiverSocketId).emit("newMessage", populatedMessage);
//         }
        
//         // Also send message back to sender to confirm
//         socket.emit("newMessage", populatedMessage);

//       } catch (error) {
//         console.error("Error saving message:", error);
//       }
//     });

//   });

//   httpServer.listen(port, () => console.log(`> Ready on http://${hostname}:${port}`));
// });