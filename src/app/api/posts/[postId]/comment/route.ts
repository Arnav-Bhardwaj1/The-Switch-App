// import { NextRequest, NextResponse } from 'next/server';
// import { getServerSession } from 'next-auth';
// import { authOptions } from '@/app/api/auth/[...nextauth]/route';
// import { connectToDB } from "@/lib/mongodb";
// import { Post } from '@/models/Post';
// import { Comment } from '@/models/Comment';
// import { User } from '@/models/User';

// export async function POST(request: NextRequest, { params }: { params: { postId: string } }) {
//   const session = await getServerSession(authOptions);
//   if (!session) return NextResponse.json({ message: 'Not Authorized' }, { status: 401 });

//   await connectToDB();
//   try {
//     const { text } = await request.json();
//     if (!text) {
//       return NextResponse.json({ message: 'Comment text is required' }, { status: 400 });
//     }

//     const newComment = new Comment({
//       post: params.postId,
//       author: session.user.id,
//       text,
//     });
//     await newComment.save();

//     // Add comment to the post's comments array
//     await Post.findByIdAndUpdate(params.postId, { $push: { comments: newComment._id } });
    
//     // Return the new comment with author details populated
//     const populatedComment = await Comment.findById(newComment._id).populate('author', 'name username profileImage');

//     return NextResponse.json(populatedComment, { status: 201 });
//   } catch (error) {
//     return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
//   }
// }

// import { NextRequest, NextResponse } from 'next/server';
// import { getServerSession } from 'next-auth';
// import { authOptions } from '@/app/api/auth/[...nextauth]/route';
// import { connectToDB } from "@/lib/mongodb";
// import { Post } from '@/models/Post';
// import { Comment } from '@/models/Comment';
// import { User } from '@/models/User';

// export async function POST(request: NextRequest, { params }: { params: { postId: string } }) {
//   const session = await getServerSession(authOptions);
//   if (!session) return NextResponse.json({ message: 'Not Authorized' }, { status: 401 });

//   await connectToDB();
//   try {
//     const { text } = await request.json();
//     if (!text) {
//       return NextResponse.json({ message: 'Comment text is required' }, { status: 400 });
//     }

//     const newComment = new Comment({
//       post: params.postId,
//       author: session.user.id,
//       text,
//     });
//     await newComment.save();

//     await Post.findByIdAndUpdate(params.postId, { $push: { comments: newComment._id } });
    
//     // Find the comment again to populate it
//     const populatedComment = await Comment.findById(newComment._id)
//       .populate({
//         path: 'author',
//         model: User,
//         select: 'name username profileImage'
//       })
//       .lean(); // ✅ ADD .lean() HERE to get a plain JavaScript object

//     // If for some reason the comment is not found after saving, handle it.
//     if (!populatedComment) {
//       return NextResponse.json({ message: 'Error retrieving new comment' }, { status: 500 });
//     }

//     return NextResponse.json(populatedComment, { status: 201 });

//   } catch (error) {
//     // This will log the specific error on the server if the problem persists
//     console.error("CRITICAL ERROR in comment API:", error); 
//     return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
//   }
// }


import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/authOptions";
import { connectToDB } from "@/lib/mongodb";
import { Post } from '@/models/Post';
import { Comment } from '@/models/Comment';
import { User } from '@/models/User';

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ postId: string }> }
) {
  const { postId } = await context.params; // Unwrap the Promise

  const session = await getServerSession(authOptions);
  if (!session) {
    return NextResponse.json({ message: 'Not Authorized' }, { status: 401 });
  }

  await connectToDB();
  try {
    const { text } = await request.json();
    if (!text) {
      return NextResponse.json({ message: 'Comment text is required' }, { status: 400 });
    }

    const newComment = new Comment({
      post: postId, // use unwrapped value
      author: session.user.id,
      text,
    });
    await newComment.save();

    await Post.findByIdAndUpdate(postId, { $push: { comments: newComment._id } });

    const populatedComment = await Comment.findById(newComment._id)
      .populate({
        path: 'author',
        model: User,
        select: 'name username profileImage',
      })
      .lean();

    if (!populatedComment) {
      throw new Error('Failed to find and populate the new comment after saving.');
    }

    return NextResponse.json(populatedComment, { status: 201 });
  } catch (error) {
    console.error('--- CRITICAL ERROR IN COMMENT API ---', error);
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}
