// import { NextRequest, NextResponse } from 'next/server';
// import { connectToDB } from "@/lib/mongodb";
// import { User } from '@/models/User';
// import { Post } from '@/models/Post';
// import { Comment } from '@/models/Comment';

// export async function GET(request: NextRequest, { params }: { params: { username: string } }) {
//   const { username } = params;

//   if (!username) {
//     return NextResponse.json({ message: 'Username is required' }, { status: 400 });
//   }

//   await connectToDB();

//   try {
//     // Find user by username, which should be unique
//     const user = await User.findOne({ username: username.toLowerCase() })
//       .select('-password')
//       .populate({
//         path: 'posts',
//         model: Post,
//         options: { sort: { createdAt: -1 } },
//         populate: [
//           { path: 'author', model: User, select: 'name username profileImage' },
//           { path: 'comments', model: Comment, populate: { path: 'author', model: User, select: 'name username profileImage' } }
//         ]
//       })
//       .populate('followers', 'name username profileImage')
//       .populate('following', 'name username profileImage');

//     if (!user) {
//       return NextResponse.json({ message: 'User not found' }, { status: 404 });
//     }

//     return NextResponse.json(user, { status: 200 });
//   } catch (error) {
//     console.error(`Error fetching user ${username}:`, error);
//     return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
//   }
// }


import { NextRequest, NextResponse } from 'next/server';
import { connectToDB } from "@/lib/mongodb";
import { User } from '@/models/User';
import { Post } from '@/models/Post';
import { Comment } from '@/models/Comment';

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ username: string }> }
) {
  const { username } = await context.params; // Await before destructuring

  if (!username) {
    return NextResponse.json({ message: 'Username is required' }, { status: 400 });
  }

  await connectToDB();

  try {
    const user = await User.findOne({ username: username.toLowerCase() })
      .select('-password')
      .populate({
        path: 'posts',
        model: Post,
        options: { sort: { createdAt: -1 } },
        populate: [
          { path: 'author', model: User, select: 'name username profileImage' },
          { path: 'comments', model: Comment, populate: { path: 'author', model: User, select: 'name username profileImage' } }
        ]
      })
      .populate('followers', 'name username profileImage')
      .populate('following', 'name username profileImage');

    if (!user) {
      return NextResponse.json({ message: 'User not found' }, { status: 404 });
    }

    return NextResponse.json(user, { status: 200 });
  } catch (error) {
    console.error(`Error fetching user ${username}:`, error);
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}
