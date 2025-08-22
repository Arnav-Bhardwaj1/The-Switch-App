import { NextRequest, NextResponse } from 'next/server';
import { connectToDB } from '@/lib/mongodb';
import { User } from '@/models/User';

export async function GET(request: NextRequest, { params }: { params: { query: string } }) {
  const { query } = params;

  if (!query) {
    return NextResponse.json({ message: 'Search query is required' }, { status: 400 });
  }

  await connectToDB();

  try {
    // Create a case-insensitive regular expression from the search query
    const searchRegex = new RegExp(query, 'i');

    const users = await User.find({
      // Search in either the username or the name field
      $or: [
        { username: { $regex: searchRegex } },
        { name: { $regex: searchRegex } },
      ],
    })
    .limit(20) // Limit the number of results to prevent overload
    .select('name username profileImage'); // Only return public, necessary data

    return NextResponse.json(users, { status: 200 });

  } catch (error) {
    console.error('Error searching for users:', error);
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}