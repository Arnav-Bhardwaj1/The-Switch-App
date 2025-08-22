import { NextResponse } from 'next/server';
import { getUploadAuthParams } from '@imagekit/next/server';

export async function GET() {
  // Return signed params for secure upload
  const { token, expire, signature } = getUploadAuthParams({
    privateKey: process.env.IMAGEKIT_PRIVATE_KEY!,
    publicKey: process.env.NEXT_PUBLIC_IMAGEKIT_PUBLIC_KEY!,
  });

  return NextResponse.json({
    token,
    expire,
    signature,
    publicKey: process.env.NEXT_PUBLIC_IMAGEKIT_PUBLIC_KEY,
  });
}
