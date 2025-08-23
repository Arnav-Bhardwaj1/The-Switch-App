// import { NextRequest, NextResponse } from 'next/server';
// import { getServerSession } from 'next-auth';
// import { authOptions } from "@/lib/authOptions";
// import { connectToDB } from "@/lib/mongodb";
// import { User } from '@/models/User';

// export async function POST(request: NextRequest, { params }: { params: { id: string } }) {
//   const session = await getServerSession(authOptions);
//   if (!session?.user?.id) {
//     return NextResponse.json({ message: 'Not Authorized' }, { status: 401 });
//   }

//   const currentUserId = session.user.id;
//   const targetUserId = params.id;

//   if (currentUserId === targetUserId) {
//     return NextResponse.json({ message: "You cannot follow yourself" }, { status: 400 });
//   }

//   await connectToDB();

//   try {
//     const currentUser = await User.findById(currentUserId);
//     const targetUser = await User.findById(targetUserId);

//     if (!currentUser || !targetUser) {
//       return NextResponse.json({ message: "User not found" }, { status: 404 });
//     }

//     const isFollowing = currentUser.following.includes(targetUserId);

//     if (isFollowing) {
//       // Unfollow logic
//       await currentUser.updateOne({ $pull: { following: targetUserId } });
//       await targetUser.updateOne({ $pull: { followers: currentUserId } });
//     } else {
//       // Follow logic
//       await currentUser.updateOne({ $push: { following: targetUserId } });
//       await targetUser.updateOne({ $push: { followers: currentUserId } });
//     }
    
//     return NextResponse.json({ success: true, isFollowing: !isFollowing }, { status: 200 });
//   } catch (error) {
//     console.error('Error handling follow action:', error);
//     return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
//   }
// }


import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/authOptions";
import { connectToDB } from "@/lib/mongodb";
import { User } from '@/models/User';

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id: targetUserId } = await context.params; // <-- unwrap Promise

  const session = await getServerSession(authOptions);
  if (!session?.user?.id) {
    return NextResponse.json({ message: 'Not Authorized' }, { status: 401 });
  }

  const currentUserId = session.user.id;

  if (currentUserId === targetUserId) {
    return NextResponse.json({ message: "You cannot follow yourself" }, { status: 400 });
  }

  await connectToDB();

  try {
    const currentUser = await User.findById(currentUserId);
    const targetUser = await User.findById(targetUserId);

    if (!currentUser || !targetUser) {
      return NextResponse.json({ message: "User not found" }, { status: 404 });
    }

    const isFollowing = currentUser.following.includes(targetUserId);

    if (isFollowing) {
      await currentUser.updateOne({ $pull: { following: targetUserId } });
      await targetUser.updateOne({ $pull: { followers: currentUserId } });
    } else {
      await currentUser.updateOne({ $push: { following: targetUserId } });
      await targetUser.updateOne({ $push: { followers: currentUserId } });
    }
    
    return NextResponse.json({ success: true, isFollowing: !isFollowing }, { status: 200 });
  } catch (error) {
    console.error('Error handling follow action:', error);
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}
