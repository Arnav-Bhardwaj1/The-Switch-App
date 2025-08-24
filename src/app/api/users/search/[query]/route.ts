import { NextRequest, NextResponse } from 'next/server';
import { connectToDB } from '@/lib/mongodb';
import { User } from '@/models/User';

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ query: string }> } // params is a Promise
) {
  const { query } = await context.params; // Await before destructuring

  if (!query) {
    return NextResponse.json({ message: 'Search query is required' }, { status: 400 });
  }

  await connectToDB();

  try {
    const searchRegex = new RegExp(query, 'i');

    const users = await User.find({
      $or: [
        { username: { $regex: searchRegex } },
        { name: { $regex: searchRegex } },
      ],
    })
      .limit(20)
      .select('name username profileImage');

    return NextResponse.json(users, { status: 200 });
  } catch (error) {
    console.error('Error searching for users:', error);
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}
