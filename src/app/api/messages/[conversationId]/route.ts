import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '../../auth/[...nextauth]/route';
import { connectToDB } from '@/lib/mongodb';
import { Message } from '@/models/Message';
import { Conversation } from '@/models/Conversation';
import { User } from '@/models/User';

export async function GET(request: NextRequest, { params }: { params: { conversationId: string } }) {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) {
    return NextResponse.json({ message: 'Not Authorized' }, { status: 401 });
  }

  await connectToDB();

  try {
    const messages = await Message.find({ conversationId: params.conversationId })
      .populate('sender', 'name username profileImage')
      .sort({ createdAt: 1 }); // Sort oldest to newest

    return NextResponse.json(messages, { status: 200 });
  } catch (error) {
    console.error('Error fetching messages:', error);
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}