import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
// It's best practice to have authOptions in a central file like /lib
import { authOptions } from "@/lib/authOptions";
import { connectToDB } from "@/lib/mongodb";
import { User } from '@/models/User';
import { Post } from '@/models/Post'; // ✅ 1. IMPORT THE POST MODEL

/**
 * @route   GET /api/users/[id]
 * @desc    Fetch a single user's profile with their posts
 * @access  Public
 */

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  if (!id) {
    return NextResponse.json({ message: 'User ID is required' }, { status: 400 });
  }

  await connectToDB();

  try {
    const user = await User.findById(id)
      .select('-password')
      .populate({
        path: 'posts',
        model: Post,
        options: { sort: { createdAt: -1 } },
        populate: {
          path: 'author',
          model: User,
          select: 'name username profileImage'
        }
      })
      .populate('followers', 'name username profileImage')
      .populate('following', 'name username profileImage');

    if (!user) {
      return NextResponse.json({ message: 'User not found' }, { status: 404 });
    }

    return NextResponse.json(user, { status: 200 });
  } catch (error) {
    console.error('Error fetching user:', error);
    if (error instanceof Error && error.name === 'CastError') {
      return NextResponse.json({ message: 'Invalid User ID format' }, { status: 400 });
    }
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}


/**
 * @route   PUT /api/users/[id]
 * @desc    Update a user's profile
 * @access  Private (Owner only)
 */

export async function PUT(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  const session = await getServerSession(authOptions);

  if (!session || session.user?.id !== id) {
    return NextResponse.json({ message: 'Not Authorized' }, { status: 403 });
  }

  await connectToDB();

  try {
    const body = await request.json();
    const { name, bio, interests, username, profileImage } = body;

    const fieldsToUpdate: Record<string, unknown> = {};
    if (name) fieldsToUpdate.name = name;
    if (bio) fieldsToUpdate.bio = bio;
    if (interests) fieldsToUpdate.interests = interests;
    if (profileImage) fieldsToUpdate.profileImage = profileImage;

    if (username) {
      const existingUser = await User.findOne({ username });
      if (existingUser && existingUser._id.toString() !== id) {
        return NextResponse.json({ message: 'Username is already taken' }, { status: 409 });
      }
      fieldsToUpdate.username = username;
    }

    const updatedUser = await User.findByIdAndUpdate(
      id,
      { $set: fieldsToUpdate },
      { new: true, runValidators: true }
    ).select('-password');

    if (!updatedUser) {
      return NextResponse.json({ message: 'User not found' }, { status: 404 });
    }

    return NextResponse.json(
      { message: 'Profile updated successfully', data: updatedUser },
      { status: 200 }
    );
  } catch (error) {
    console.error('Error updating user:', error);
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}
