// import { NextRequest, NextResponse } from 'next/server';
// import { getServerSession } from 'next-auth';
// import { authOptions } from "@/lib/authOptions";
// import { connectToDB } from "@/lib/mongodb";
// import { Post } from '@/models/Post';

// export async function POST(request: NextRequest, { params }: { params: { postId: string } }) {
//   const session = await getServerSession(authOptions);
//   if (!session) return NextResponse.json({ message: 'Not Authorized' }, { status: 401 });

//   await connectToDB();
//   try {
//     const post = await Post.findById(params.postId);
//     if (!post) return NextResponse.json({ message: 'Post not found' }, { status: 404 });

//     const userId = session.user.id;
//     const userIndex = post.likes.indexOf(userId);

//     if (userIndex > -1) {
//       // User has already liked, so unlike
//       post.likes.pull(userId);
//     } else {
//       // User has not liked, so like
//       post.likes.push(userId);
//     }

//     await post.save();
//     return NextResponse.json({ likes: post.likes }, { status: 200 });
//   } catch {
//     return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
//   }
// }

import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/authOptions";
import { connectToDB } from "@/lib/mongodb";
import { Post } from '@/models/Post';

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ postId: string }> }
) {
  const { postId } = await context.params; // unwrap Promise

  const session = await getServerSession(authOptions);
  if (!session) return NextResponse.json({ message: 'Not Authorized' }, { status: 401 });

  await connectToDB();
  try {
    const post = await Post.findById(postId);
    if (!post) return NextResponse.json({ message: 'Post not found' }, { status: 404 });

    const userId = session.user.id;
    const userIndex = post.likes.indexOf(userId);

    if (userIndex > -1) {
      post.likes.pull(userId); // unlike
    } else {
      post.likes.push(userId); // like
    }

    await post.save();
    return NextResponse.json({ likes: post.likes }, { status: 200 });
  } catch {
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}
