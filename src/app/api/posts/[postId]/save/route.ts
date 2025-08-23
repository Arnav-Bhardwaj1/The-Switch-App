// import { NextRequest, NextResponse } from 'next/server';
// import { getServerSession } from 'next-auth';
// import { authOptions } from "@/lib/authOptions";
// import { connectToDB } from "@/lib/mongodb";
// import { Post } from '@/models/Post';
// import { User } from '@/models/User';

// export async function POST(request: NextRequest, { params }: { params: { postId: string } }) {
//   const session = await getServerSession(authOptions);
//   if (!session) return NextResponse.json({ message: 'Not Authorized' }, { status: 401 });

//   await connectToDB();
//   try {
//     const userId = session.user.id;
//     const postId = params.postId;

//     const user = await User.findById(userId);
//     const post = await Post.findById(postId);

//     if (!user || !post) return NextResponse.json({ message: 'User or Post not found' }, { status: 404 });

//     const isSaved = user.savedPosts.includes(postId);

//     if (isSaved) {
//       // Unsave
//       await User.findByIdAndUpdate(userId, { $pull: { savedPosts: postId } });
//       await Post.findByIdAndUpdate(postId, { $pull: { savedBy: userId } });
//     } else {
//       // Save
//       await User.findByIdAndUpdate(userId, { $push: { savedPosts: postId } });
//       await Post.findByIdAndUpdate(postId, { $push: { savedBy: userId } });
//     }

//     return NextResponse.json({ message: 'Success' }, { status: 200 });
//   } catch {
//     return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
//   }
// }

import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/authOptions";
import { connectToDB } from "@/lib/mongodb";
import { Post } from '@/models/Post';
import { User } from '@/models/User';

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ postId: string }> }
) {
  const { postId } = await context.params; // <-- unwrap Promise

  const session = await getServerSession(authOptions);
  if (!session) return NextResponse.json({ message: 'Not Authorized' }, { status: 401 });

  await connectToDB();
  try {
    const userId = session.user.id;

    const user = await User.findById(userId);
    const post = await Post.findById(postId);

    if (!user || !post) {
      return NextResponse.json({ message: 'User or Post not found' }, { status: 404 });
    }

    const isSaved = user.savedPosts.includes(postId);

    if (isSaved) {
      await User.findByIdAndUpdate(userId, { $pull: { savedPosts: postId } });
      await Post.findByIdAndUpdate(postId, { $pull: { savedBy: userId } });
    } else {
      await User.findByIdAndUpdate(userId, { $push: { savedPosts: postId } });
      await Post.findByIdAndUpdate(postId, { $push: { savedBy: userId } });
    }

    return NextResponse.json({ message: 'Success' }, { status: 200 });
  } catch {
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}
