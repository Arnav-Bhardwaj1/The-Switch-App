// import NextAuth, { NextAuthOptions } from "next-auth";
// import CredentialsProvider from "next-auth/providers/credentials";
// import { JWT } from "next-auth/jwt";
// import { Session } from "next-auth";
// import { connectToDB } from "@/lib/mongodb";
// import { User } from "@/models/User";
// import bcrypt from "bcryptjs";

// export const authOptions: NextAuthOptions = {
//     providers: [
//         CredentialsProvider({
//             name: "Credentials",
//             credentials: {
//                 email: { label: "Email", type: "text" },
//                 password: { label: "Password", type: "password" },
//             },
//             async authorize(credentials) {
//                 if (!credentials?.email || !credentials?.password) {
//                     throw new Error("Missing credentials");
//                 }

//                 await connectToDB();

//                 const user = await User.findOne({ email: credentials.email });
//                 if (!user) throw new Error("User not found");

//                 const isValid = await bcrypt.compare(credentials.password, user.password);
//                 if (!isValid) throw new Error("Invalid password");

//                 // --- FIX 1: Return the gender from the user document ---
//                 return {
//                     id: user._id.toString(),
//                     email: user.email,
//                     name: user.name,
//                     gender: user.gender, // <-- ADD THIS LINE
//                     username: user.username, // ✅ Add username
//                 };
//             },
//         }),
//     ],
//     callbacks: {
//         // --- FIX 2: Add the jwt callback ---
//         async jwt({ token, user }: { token: JWT; user?: { id: string; email: string; name: string; gender: string; username: string } }) {
//             // On sign in, `user` object is available. Persist the custom data to the token.
//             if (user) {
//                 token.id = user.id;
//                 token.gender = user.gender; // <-- Add user's gender to the token
//                 token.username = user.username; // ✅ Add username to the token

//             }
//             return token;
//         },
//         // --- FIX 3: Update the session callback ---
//         async session({ session, token }: { session: Session; token: JWT }) {
//             // Add the custom data from the token to the session object
//             if (session.user) {
//                 session.user.id = token.id as string;
//                 session.user.gender = token.gender as string; // <-- Add gender to the session
//                 session.user.username = token.username as string; // ✅ Add username to the session
//             }
//             return session;
//         },
//     },
//     session: {
//         strategy: 'jwt' as const,
//     },
//     pages: {
//         signIn: "/auth/login",
//     },
//     secret: process.env.NEXTAUTH_SECRET,
// };

// const handler = NextAuth(authOptions);
// export { handler as GET, handler as POST };


// import NextAuth, { NextAuthOptions } from "next-auth";
// import CredentialsProvider from "next-auth/providers/credentials";
// import { JWT } from "next-auth/jwt";
// import { Session } from "next-auth";
// import { connectToDB } from "@/lib/mongodb";
// import { User } from "@/models/User";
// import bcrypt from "bcryptjs";

// export const authOptions: NextAuthOptions = {
//   providers: [
//     CredentialsProvider({
//       name: "Credentials",
//       credentials: {
//         email: { label: "Email", type: "text" },
//         password: { label: "Password", type: "password" },
//       },
//       async authorize(credentials) {
//         if (!credentials?.email || !credentials?.password) {
//           throw new Error("Missing credentials");
//         }

//         await connectToDB();
//         const user = await User.findOne({ email: credentials.email });
//         if (!user) throw new Error("User not found");

//         const isValid = await bcrypt.compare(credentials.password, user.password);
//         if (!isValid) throw new Error("Invalid password");

//         return {
//           id: user._id.toString(),
//           email: user.email,
//           name: user.name,
//           gender: user.gender,
//           username: user.username,
//         };
//       },
//     }),
//   ],
//   callbacks: {
//     async jwt({ token, user }: { token: JWT; user?: { id: string; email: string; name: string; gender: string; username: string } }) {
//       if (user) {
//         token.id = user.id;
//         token.gender = user.gender;
//         token.username = user.username;
//       }
//       return token;
//     },
//     async session({ session, token }: { session: Session; token: JWT }) {
//       if (session.user) {
//         session.user.id = token.id as string;
//         session.user.gender = token.gender as string;
//         session.user.username = token.username as string;
//       }
//       return session;
//     },
//   },
//   session: {
//     strategy: "jwt",
//   },
//   pages: {
//     signIn: "/auth/login",
//   },
//   secret: process.env.NEXTAUTH_SECRET,
// };

// const handler = NextAuth(authOptions);
// export { handler as GET, handler as POST };



import NextAuth from "next-auth";
import { authOptions } from "@/lib/authOptions";

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
