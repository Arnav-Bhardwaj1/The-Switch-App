// import { NextRequest, NextResponse } from 'next/server';
// import { getServerSession } from 'next-auth';
// import { authOptions } from '../auth/[...nextauth]/route';
// import { connectToDB } from '@/lib/mongodb';
// import { Post } from '@/models/Post';
// import { User } from '@/models/User';
// import { Comment } from '@/models/Comment'; // ✅ THIS IMPORT FIXES THE ERROR


// export async function POST(request: NextRequest) {
//   // 1. Authenticate the user
//   const session = await getServerSession(authOptions);
//   if (!session || !session.user) {
//     return NextResponse.json({ message: 'Not Authorized' }, { status: 401 });
//   }

//   await connectToDB();

//   try {
//     const body = await request.json();
//     const { text, photo, video, topics, tags } = body;
//     const authorId = session.user.id;
    
//     // Basic validation
//     if (!text && !photo && !video) {
//         return NextResponse.json({ message: 'Post must contain text, a photo, or a video.' }, { status: 400 });
//     }

//     // 2. Create the new post document
//     const newPost = new Post({
//       author: authorId,
//       text,
//       photo,
//       video,
//       topics: topics || [],
//       tags: tags || [],
//     });

//     await newPost.save();

//     // 3. Add the new post's ID to the user's 'posts' array
//     await User.findByIdAndUpdate(
//       authorId,
//       { $push: { posts: newPost._id } },
//       { new: true }
//     );

//     // 4. Populate author details for the response
//     const populatedPost = await Post.findById(newPost._id).populate('author', 'name username profileImage');

//     return NextResponse.json(populatedPost, { status: 201 });

//   } catch (error) {
//     console.error('Error creating post:', error);
//     return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
//   }
// }


// export async function GET(request: NextRequest) {
//   console.log("--- GET /api/posts: REQUEST RECEIVED ---");
//   await connectToDB();

//   try {
//     console.log("Step 1: Building the query to find and sort posts...");
//     const posts = await Post.find({})
//       .sort({ createdAt: -1 }) // Sort by newest first
//       .populate([
//         { 
//           path: 'author', 
//           model: User, 
//           // ✅ Added 'gender' to the selection
//           select: 'name username profileImage gender' 
//         },
//         { 
//           path: 'comments', 
//           model: Comment,
//           populate: { 
//             path: 'author', 
//             model: User, 
//             select: 'name username profileImage' 
//           }
//         }
//       ])
//       .lean();
    
//     console.log(`Step 2: Query successful. Found ${posts.length} posts.`);
//     return NextResponse.json(posts, { status: 200 });

//   } catch (error) {
//     // ❗ THIS WILL PRINT THE REAL ERROR TO YOUR SERVER TERMINAL
//     console.error('--- CRITICAL ERROR fetching posts ---', error);
//     return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
//   }
// }

import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/authOptions";
import { connectToDB } from '@/lib/mongodb';
import { Post } from '@/models/Post';
import { User } from '@/models/User';
import { Comment } from '@/models/Comment';

export async function POST(request: NextRequest) {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id || !session.user.gender) {
    return NextResponse.json({ message: 'Not Authorized or session is missing user data' }, { status: 401 });
  }

  await connectToDB();

  try {
    const body = await request.json();
    
    // ✅ DEBUG: Log the entire body received from the frontend
    console.log("--- RECEIVED REQUEST BODY ---", body);

    const { text, photo, video, topics, tags, space } = body;
    const authorId = session.user.id;
    const authorGender = session.user.gender;
    
    if (!text && !photo && !video) {
        return NextResponse.json({ message: 'Post must contain text, a photo, or a video.' }, { status: 400 });
    }

    const requestedSpace = space || 'common';
    if (requestedSpace !== 'common' && requestedSpace !== authorGender) {
        return NextResponse.json({ message: `You are not permitted to post in the '${requestedSpace}' space.` }, { status: 403 });
    }

    const postDataToSave = {
      author: authorId,
      text,
      photo,
      video,
      topics: topics || [],
      tags: tags || [],
      space: requestedSpace,
      // space: 'male',
    };

    // ✅ DEBUG: Log the exact object we are about to save
    console.log("--- DATA TO BE SAVED ---", postDataToSave);

    const newPost = new Post(postDataToSave);
    await newPost.save();

    await User.findByIdAndUpdate(
      authorId,
      { $push: { posts: newPost._id } },
      { new: true }
    );

    const populatedPost = await Post.findById(newPost._id).populate('author', 'name username profileImage gender').lean();

    return NextResponse.json(populatedPost, { status: 201 });

  } catch (error) {
    console.error('Error creating post:', error);
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}

// GET handler remains the same
export async function GET() {
  await connectToDB();
  try {
    const posts = await Post.find({})
      .sort({ createdAt: -1 })
      .populate([
        { path: 'author', model: User, select: 'name username profileImage gender' },
        { path: 'comments', model: Comment, populate: { path: 'author', model: User, select: 'name username profileImage' } }
      ])
      .lean();
    return NextResponse.json(posts, { status: 200 });
  } catch (error) {
    console.error('--- CRITICAL ERROR fetching posts ---', error);
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}