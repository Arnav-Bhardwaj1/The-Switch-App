import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/authOptions";
import { connectToDB } from "@/lib/mongodb";
import { User } from '@/models/User';

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) {
    // Return empty array if not logged in
    return NextResponse.json([], { status: 200 });
  }

  await connectToDB();

  try {
    const currentUser = await User.findById(session.user.id).select('interests following');
    if (!currentUser) {
      return NextResponse.json([], { status: 200 });
    }

    const suggestions = await User.find({
      // Conditions for suggestions:
      _id: { 
        $ne: currentUser._id, // Not the current user
        $nin: currentUser.following // Not anyone the user already follows
      },
      // And has at least one interest in common
      interests: { $in: currentUser.interests }
    })
    .limit(5) // Suggest up to 5 users
    .select('name username profileImage'); // Only send public, necessary data

    return NextResponse.json(suggestions, { status: 200 });

  } catch (error) {
    console.error('Error fetching user suggestions:', error);
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}