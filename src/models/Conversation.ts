// import mongoose, { Schema, model, models } from 'mongoose';

// const ConversationSchema = new Schema({
//   participants: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],
//   messages: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Message' }],
// }, { timestamps: true });

// export const Conversation = models.Conversation || model('Conversation', ConversationSchema);
import mongoose, { Schema, model, models } from 'mongoose';

const ConversationSchema = new Schema({
  participants: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }], // Should ref 'User'
  messages: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Message' }], // Should ref 'Message'
}, { timestamps: true });

export const Conversation = models.Conversation || model('Conversation', ConversationSchema);