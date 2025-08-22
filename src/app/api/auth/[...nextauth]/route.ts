import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import { connectToDB } from "@/lib/mongodb";
import { User } from "@/models/User";
import bcrypt from "bcryptjs";

export const authOptions = {
    providers: [
        CredentialsProvider({
            name: "Credentials",
            credentials: {
                email: { label: "Email", type: "text" },
                password: { label: "Password", type: "password" },
            },
            async authorize(credentials) {
                if (!credentials?.email || !credentials?.password) {
                    throw new Error("Missing credentials");
                }

                await connectToDB();

                const user = await User.findOne({ email: credentials.email });
                if (!user) throw new Error("User not found");

                const isValid = await bcrypt.compare(credentials.password, user.password);
                if (!isValid) throw new Error("Invalid password");

                // --- FIX 1: Return the gender from the user document ---
                return {
                    id: user._id.toString(),
                    email: user.email,
                    name: user.name,
                    gender: user.gender, // <-- ADD THIS LINE
                    username: user.username, // ✅ Add username
                };
            },
        }),
    ],
    callbacks: {
        // --- FIX 2: Add the jwt callback ---
        async jwt({ token, user }: { token: any, user: any }) {
            // On sign in, `user` object is available. Persist the custom data to the token.
            if (user) {
                token.id = user.id;
                token.gender = user.gender; // <-- Add user's gender to the token
                token.username = user.username; // ✅ Add username to the token

            }
            return token;
        },
        // --- FIX 3: Update the session callback ---
        async session({ session, token }: { session: any, token: any }) {
            // Add the custom data from the token to the session object
            if (session.user) {
                session.user.id = token.id;
                session.user.gender = token.gender; // <-- Add gender to the session
                session.user.username = token.username as string; // ✅ Add username to the session
            }
            return session;
        },
    },
    session: {
        strategy: 'jwt' as const,
    },
    pages: {
        signIn: "/auth/login",
    },
    secret: process.env.NEXTAUTH_SECRET,
};

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };