// 'use client'

// import { useState } from 'react'
// import { useRouter } from 'next/navigation'
// import { toast } from 'sonner'
// import {
//   upload,
//   ImageKitAbortError,
//   ImageKitInvalidRequestError,
//   ImageKitServerError,
//   ImageKitUploadNetworkError,
// } from '@imagekit/next'

// import { Input } from '@/components/ui/input'
// import { Button } from '@/components/ui/button'
// import {
//   Select,
//   SelectContent,
//   SelectItem,
//   SelectTrigger,
//   SelectValue,
// } from '@/components/ui/select'
// import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
// import { Label } from '@/components/ui/label'
// import { Textarea } from '@/components/ui/textarea'
// import { Progress } from '@/components/ui/progress' // Optional: for a nice progress bar

// export default function SignupPage() {
//   const [form, setForm] = useState({
//     name: '',
//     username: '',
//     email: '',
//     password: '',
//     gender: 'male',
//     bio: '',
//     profileImage: '',
//   })

//   const [loading, setLoading] = useState(false)
//   const [file, setFile] = useState<File | null>(null)
//   const [uploadProgress, setUploadProgress] = useState(0)
//   const router = useRouter()

//   // 1. Fetch secure authentication parameters from your server
//   const getAuthParams = async () => {
//     try {
//       const res = await fetch('/api/imagekit-auth')
//       if (!res.ok) {
//         throw new Error(`Failed to get auth params: ${res.statusText}`)
//       }
//       return res.json()
//     } catch (error) {
//       console.error('Auth error', error)
//       toast.error('Could not authenticate with image service.')
//       return null
//     }
//   }

//   // 2. Handle the secure upload using the @imagekit/next package
//   const handleUpload = async () => {
//     if (!file) return ''
//     setUploadProgress(0)

//     const authParams = await getAuthParams()
//     if (!authParams) return ''

//     try {
//       const response = await upload({
//         file,
//         fileName: file.name,
//         ...authParams,
//         onProgress: (e) => {
//           setUploadProgress((e.loaded / e.total) * 100)
//         },
//       })
//       toast.success('🖼️ Image uploaded!')
//       return response.url
//     } catch (error) {
//       if (error instanceof ImageKitAbortError) {
//         toast.error(`Upload aborted: ${error.reason}`)
//       } else if (error instanceof ImageKitInvalidRequestError) {
//         toast.error(`Invalid request: ${error.message}`)
//       } else if (error instanceof ImageKitUploadNetworkError) {
//         toast.error(`Network error: ${error.message}`)
//       } else if (error instanceof ImageKitServerError) {
//         toast.error(`Server error: ${error.message}`)
//       } else {
//         toast.error('An unknown upload error occurred.')
//       }
//       console.error(error)
//       return '' // Return empty string on failure
//     }
//   }

//   // 3. Main form submission handler
//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault()
//     setLoading(true)

//     try {
//       let uploadedUrl = ''
//       if (file) {
//         uploadedUrl = await handleUpload()
//         // If upload fails, uploadedUrl will be empty. Stop submission.
//         if (!uploadedUrl) {
//           throw new Error('Image upload failed, please try again.')
//         }
//       }

//       const res = await fetch('/api/signup', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({
//           ...form,
//           profileImage: uploadedUrl || '', // Use uploaded URL
//         }),
//       })

//       if (res.status === 201) {
//         toast.success('🎉 Account created! Redirecting to login...')
//         setTimeout(() => router.push('/auth/login'), 1500)
//       } else {
//         const errMsg = await res.text()
//         toast.error(errMsg || '❌ Something went wrong.')
//       }
//     } catch (err: any) {
//       // Catch errors from both upload and signup fetch
//       toast.error(err.message || '⚠️ Network error.')
//     } finally {
//       setLoading(false)
//       setUploadProgress(0) // Reset progress
//     }
//   }

//   const getButtonText = () => {
//     if (loading) {
//       if (uploadProgress > 0 && uploadProgress < 100) {
//         return `Uploading: ${Math.round(uploadProgress)}%`
//       }
//       return 'Creating Account...'
//     }
//     return 'Sign Up'
//   }

//   return (
//     <div className="min-h-screen flex items-center justify-center bg-muted px-4 py-8">
//       <Card className="w-full max-w-md">
//         <CardHeader>
//           <CardTitle className="text-2xl text-center">Create Account</CardTitle>
//         </CardHeader>
//         <CardContent>
//           <form onSubmit={handleSubmit} className="space-y-4">
//             {/* ... other form fields (name, username, email, etc.) remain the same ... */}
//             <div className="grid gap-1">
//               <Label htmlFor="name">Full Name</Label>
//               <Input
//                 id="name"
//                 required
//                 value={form.name}
//                 onChange={(e) => setForm({ ...form, name: e.target.value })}
//               />
//             </div>

//             <div className="grid gap-1">
//               <Label htmlFor="username">Username</Label>
//               <Input
//                 id="username"
//                 required
//                 value={form.username}
//                 onChange={(e) => setForm({ ...form, username: e.target.value })}
//               />
//             </div>

//             <div className="grid gap-1">
//               <Label htmlFor="email">Email</Label>
//               <Input
//                 id="email"
//                 type="email"
//                 required
//                 value={form.email}
//                 onChange={(e) => setForm({ ...form, email: e.target.value })}
//               />
//             </div>

//             <div className="grid gap-1">
//               <Label htmlFor="password">Password</Label>
//               <Input
//                 id="password"
//                 type="password"
//                 required
//                 value={form.password}
//                 onChange={(e) => setForm({ ...form, password: e.target.value })}
//               />
//             </div>

//             <div className="grid gap-1">
//               <Label htmlFor="gender">Gender</Label>
//               <Select
//                 value={form.gender}
//                 onValueChange={(val) => setForm({ ...form, gender: val })}
//               >
//                 <SelectTrigger>
//                   <SelectValue placeholder="Select gender" />
//                 </SelectTrigger>
//                 <SelectContent>
//                   <SelectItem value="male">Male</SelectItem>
//                   <SelectItem value="female">Female</SelectItem>
//                   <SelectItem value="trans">Transgender</SelectItem>
//                   <SelectItem value="other">Other</SelectItem>
//                 </SelectContent>
//               </Select>
//             </div>

//             <div className="grid gap-1">
//               <Label htmlFor="bio">Bio (optional)</Label>
//               <Textarea
//                 id="bio"
//                 value={form.bio}
//                 onChange={(e) => setForm({ ...form, bio: e.target.value })}
//               />
//             </div>

//             <div className="grid gap-1">
//               <Label htmlFor="profileImage">Profile Image / Video</Label>
//               <Input
//                 id="profileImage"
//                 type="file"
//                 accept="image/*,video/*"
//                 onChange={(e) => {
//                   if (e.target.files?.[0]) {
//                     setFile(e.target.files[0])
//                     setUploadProgress(0) // Reset progress on new file select
//                   }
//                 }}
//                 disabled={loading}
//               />
//             </div>
            
//             {loading && uploadProgress > 0 && (
//               <Progress value={uploadProgress} className="w-full" />
//             )}

//             <Button type="submit" className="w-full" disabled={loading}>
//               {getButtonText()}
//             </Button>
//           </form>

//           <p className="text-center text-sm mt-4 text-muted-foreground">
//             Already have an account?{' '}
//             <a href="/auth/login" className="text-primary underline">
//               Login
//             </a>
//           </p>
//         </CardContent>
//       </Card>
//     </div>
//   )
// }


'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { toast } from 'sonner'
import {
  upload,
  ImageKitAbortError,
  ImageKitInvalidRequestError,
  ImageKitServerError,
  ImageKitUploadNetworkError,
} from '@imagekit/next'

import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Progress } from '@/components/ui/progress'

export default function SignupPage() {
  const [form, setForm] = useState({
    name: '',
    username: '',
    email: '',
    password: '',
    gender: 'male',
    bio: '',
    profileImage: '',
  })
  const [otp, setOtp] = useState('');
  const [showOtpForm, setShowOtpForm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const router = useRouter();

  // 1. Fetch secure authentication parameters from your server
  const getAuthParams = async () => {
    try {
      const res = await fetch('/api/imagekit-auth')
      if (!res.ok) {
        throw new Error(`Failed to get auth params: ${res.statusText}`)
      }
      return res.json()
    } catch (error) {
      console.error('Auth error', error)
      toast.error('Could not authenticate with image service.')
      return null
    }
  }

  // 2. Handle the secure upload using the @imagekit/next package
  const handleUpload = async () => {
    if (!file) return ''
    setUploadProgress(0)

    const authParams = await getAuthParams()
    if (!authParams) return ''

    try {
      const response = await upload({
        file,
        fileName: file.name,
        ...authParams,
        onProgress: (e) => {
          setUploadProgress((e.loaded / e.total) * 100)
        },
      })
      toast.success('🖼️ Image uploaded!')
      return response.url
    } catch (error) {
      if (error instanceof ImageKitAbortError) {
        toast.error(`Upload aborted: ${error.reason}`)
      } else if (error instanceof ImageKitInvalidRequestError) {
        toast.error(`Invalid request: ${error.message}`)
      } else if (error instanceof ImageKitUploadNetworkError) {
        toast.error(`Network error: ${error.message}`)
      } else if (error instanceof ImageKitServerError) {
        toast.error(`Server error: ${error.message}`)
      } else {
        toast.error('An unknown upload error occurred.')
      }
      console.error(error)
      return '' // Return empty string on failure
    }
  }

  // --- Main Form Submission Handler ---
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      let uploadedUrl = '';
      if (file) {
        uploadedUrl = await handleUpload();
        if (!uploadedUrl) {
          throw new Error('Image upload failed, please try again.');
        }
      }

      const res = await fetch('/api/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          profileImage: uploadedUrl || '',
        }),
      });
      
      const data = await res.json();

      if (res.status === 201) {
        toast.success(data.message);
        setShowOtpForm(true); // Show OTP form on success
      } else {
        toast.error(data.error || '❌ Something went wrong.');
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : '⚠️ Network error.';
      toast.error(errorMessage);
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };
  
  // --- OTP Verification Handler ---
  const handleOtpVerification = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
        const res = await fetch('/api/verify-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: form.email, otp }),
        });

        const data = await res.json();

        if (res.ok) {
            toast.success('✅ Email verified! Redirecting to login...');
            setTimeout(() => router.push('/auth/login'), 1500);
        } else {
            toast.error(data.error || 'Verification failed.');
        }
    } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : '⚠️ Network error.';
        toast.error(errorMessage);
    } finally {
        setLoading(false);
    }
  }

  const getButtonText = () => {
    if (loading) {
      if (showOtpForm) return 'Verifying...';
      if (uploadProgress > 0 && uploadProgress < 100) {
        return `Uploading: ${Math.round(uploadProgress)}%`;
      }
      return 'Sending OTP...';
    }
    return showOtpForm ? 'Verify & Sign Up' : 'Get OTP';
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-muted px-4 py-8">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-2xl text-center">
            {showOtpForm ? 'Verify Your Email' : 'Create Account'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!showOtpForm ? (
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* --- All original form fields --- */}
              <div className="grid gap-1">
                <Label htmlFor="name">Full Name</Label>
                <Input id="name" required value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
              </div>
              <div className="grid gap-1">
                <Label htmlFor="username">Username</Label>
                <Input id="username" required value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} />
              </div>
              <div className="grid gap-1">
                <Label htmlFor="email">Email</Label>
                <Input id="email" type="email" required value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
              </div>
              <div className="grid gap-1">
                <Label htmlFor="password">Password</Label>
                <Input id="password" type="password" required value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
              </div>
              <div className="grid gap-1">
                <Label htmlFor="gender">Gender</Label>
                <Select value={form.gender} onValueChange={(val) => setForm({ ...form, gender: val })}>
                  <SelectTrigger><SelectValue placeholder="Select gender" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="male">Male</SelectItem>
                    <SelectItem value="female">Female</SelectItem>
                    <SelectItem value="trans">Transgender</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-1">
                <Label htmlFor="bio">Bio (optional)</Label>
                <Textarea id="bio" value={form.bio} onChange={(e) => setForm({ ...form, bio: e.target.value })} />
              </div>
              <div className="grid gap-1">
                <Label htmlFor="profileImage">Profile Image / Video</Label>
                <Input id="profileImage" type="file" accept="image/*,video/*" onChange={(e) => { if (e.target.files?.[0]) { setFile(e.target.files[0]); setUploadProgress(0); } }} disabled={loading} />
              </div>
              {loading && uploadProgress > 0 && (
                <Progress value={uploadProgress} className="w-full" />
              )}
              <Button type="submit" className="w-full" disabled={loading}>
                {getButtonText()}
              </Button>
            </form>
          ) : (
            <form onSubmit={handleOtpVerification} className="space-y-4">
               <p className="text-center text-sm text-muted-foreground">
                An OTP has been sent to <strong>{form.email}</strong>. Please enter it below.
              </p>
              <div className="grid gap-1">
                <Label htmlFor="otp">One-Time Password (OTP)</Label>
                <Input
                  id="otp"
                  required
                  value={otp}
                  onChange={(e) => setOtp(e.target.value)}
                  placeholder="Enter 6-digit OTP"
                />
              </div>
              <Button type="submit" className="w-full" disabled={loading}>
                {getButtonText()}
              </Button>
            </form>
          )}

          <p className="text-center text-sm mt-4 text-muted-foreground">
            Already have an account?{' '}
            <a href="/auth/login" className="text-primary underline">
              Login
            </a>
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
