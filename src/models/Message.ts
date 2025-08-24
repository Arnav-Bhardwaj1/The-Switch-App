import mongoose, { Schema, model, models } from 'mongoose';

const MessageSchema = new Schema({
  conversationId: { type: mongoose.Schema.Types.ObjectId, ref: 'Conversation' }, // Should ref 'Conversation'
  sender: { type: mongoose.Schema.Types.ObjectId, ref: 'User' }, // Should ref 'User'
  text: { type: String, required: true },
}, { timestamps: true });

export const Message = models.Message || model('Message', MessageSchema);