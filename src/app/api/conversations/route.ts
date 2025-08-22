import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '../auth/[...nextauth]/route';
import { connectToDB } from '@/lib/mongodb';
import { User } from '@/models/User';
import { Conversation } from '@/models/Conversation';
import { Message } from '@/models/Message';

export async function GET(request: NextRequest) {
    const session = await getServerSession(authOptions);
    if (!session?.user?.id) {
        return NextResponse.json({ message: 'Not Authorized' }, { status: 401 });
    }

    await connectToDB();
    try {
        const conversations = await Conversation.find({ participants: session.user.id })
            .populate('participants', 'name username profileImage')
            .populate({
                path: 'messages',
                model: Message,
                perDocumentLimit: 1, // Get only the last message
                options: { sort: { createdAt: -1 } }
            })
            .sort({ updatedAt: -1 });

        return NextResponse.json(conversations, { status: 200 });
    } catch (error) {
        console.error("--- CRITICAL ERROR fetching conversations ---", error);

        return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
    }
}