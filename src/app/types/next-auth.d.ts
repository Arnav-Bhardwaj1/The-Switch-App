import NextAuth from "next-auth";

declare module "next-auth" {
  interface Session {
    user: {
      id: string;
      name?: string | null;
      email?: string | null;
      image?: string | null;
      gender?: "male" | "female" | "other"; // or your Gender enum/type
      username?: string;
    };
  }

  interface User {
    id: string;
    gender?: "male" | "female" | "other"; // match your User model
    username?: string;
  }
  interface Post {
  _id: string;
  text: string;
  photo?: string;
  video?: string;
  createdAt: string;
  author: Author;
  likes: string[];
  savedBy: string[];
  comments: Comment[];
  topics?: string[]; // add optional fields
  tags?: string[];
}

}
