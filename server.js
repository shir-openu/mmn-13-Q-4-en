// Simple local development server
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenerativeAI } from '@google/generative-ai';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;
const MAX_ATTEMPTS = 10;

// MIME types
const mimeTypes = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon'
};

// Helper function to call OpenRouter API
async function callOpenRouter(prompt) {
  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'openai/gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }]
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenRouter API error: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

// Helper function to call Google Gemini API
async function callGemini(prompt) {
  const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
  const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

  const result = await model.generateContent(prompt);
  const response = await result.response;
  return response.text();
}

// AI Handler function
async function handleAIRequest(body) {
  const { userInput, currentStep, problemData, conversationHistory } = body;

  // Check attempt limit
  if (conversationHistory && conversationHistory.length >= MAX_ATTEMPTS) {
    return {
      hint: `You have used all ${MAX_ATTEMPTS} attempts. Here is the full solution:\n\n` + problemData.fullSolution
    };
  }

  // Determine which AI provider to use
  const aiProvider = process.env.AI_PROVIDER || 'google';

  // Build conversation history text
  let conversationText = '';
  if (conversationHistory && conversationHistory.length > 0) {
    conversationHistory.forEach(turn => {
      conversationText += `Student answer: ${turn.user}\nTeacher response: ${turn.ai}\n\n`;
    });
  }

  const prompt = `
# CRITICAL INSTRUCTIONS
1. Respond in ENGLISH only
2. Be PRACTICAL and SPECIFIC - give concrete mathematical guidance
3. Keep responses 2-4 sentences
4. Use clear, accessible language
5. NEVER give the complete final answer until ${MAX_ATTEMPTS} attempts exhausted
6. NEVER repeat the same hint - check conversation history and progress
7. NEVER put quotes around equations - write them directly without '' or "" marks
8. ACCEPT ANY MATHEMATICALLY EQUIVALENT FORM of the correct answer

# MATHEMATICAL EQUIVALENCE FOR DIFFERENTIAL EQUATIONS
When comparing student answers to the expected solution:
- Arbitrary constants (A, B, C, etc.) are ARBITRARY by definition
- Different constant names represent the same family of solutions
- Constant coefficients are irrelevant (constants can absorb any scalar multiple)
- What matters is the FUNCTIONAL FORM (exponentials, polynomials, structure)
- If the student's solution represents the same family of functions, it is CORRECT
- Focus on whether the solution satisfies the differential equation, not constant notation

---

# The Exercise: Triangular ODE System - Substitution Method

## The System:
dx/dt = 5x + 2y
dy/dt = 5y + e^{5t}  ← Start here (depends only on y)
dz/dt = 3y + 5z

## COMPLETE SOLUTIONS (your reference):

**Step 1 - Solve y(t):**
- Equation: y' - 5y = e^{5t} (first-order linear)
- Integrating factor: μ = e^{-5t}
- Multiply both sides: e^{-5t}y' - 5e^{-5t}y = 1
- Left side is derivative: (e^{-5t}y)' = 1
- Integrate: e^{-5t}y = t + B
- ANSWER: y(t) = te^{5t} + Be^{5t} = (t+B)e^{5t}

**Step 2 - Solve x(t):**
- Substitute y: x' - 5x = 2y = 2te^{5t} + 2Be^{5t}
- Integrating factor: μ = e^{-5t}
- (e^{-5t}x)' = 2t + 2B
- Integrate: e^{-5t}x = t² + 2Bt + A
- ANSWER: x(t) = t²e^{5t} + 2Bte^{5t} + Ae^{5t}

**Step 3 - Solve z(t):**
- Substitute y: z' - 5z = 3y = 3te^{5t} + 3Be^{5t}
- Integrating factor: μ = e^{-5t}
- (e^{-5t}z)' = 3t + 3B
- Integrate: e^{-5t}z = (3/2)t² + 3Bt + C
- ANSWER: z(t) = (3/2)t²e^{5t} + 3Bte^{5t} + Ce^{5t}

---

## Current Step: ${currentStep}
## Expected Answer: ${problemData.correctAnswer}
## Student Input: ${userInput}

${conversationText ? `## Previous Conversation:\n${conversationText}` : ''}

---

# SPECIFIC HINTS BY STEP (give progressively):

## If Step 1 (solving y):
- Hint 1: "The equation y' - 5y = e^{5t} is a first-order linear ODE. Use the integrating factor method."
- Hint 2: "The integrating factor is μ = e^{-5t}. Multiply both sides by this factor."
- Hint 3: "After multiplying, the left side becomes a derivative: (e^{-5t}·y)' = 1"
- Hint 4: "Integrate both sides: e^{-5t}·y = t + B. Now isolate y."

## If Step 2 (solving x):
- Hint 1: "Substitute the y you found into the first equation. You get: x' - 5x = 2y = 2(t+B)e^{5t}"
- Hint 2: "This is again a linear ODE with the same integrating factor: μ = e^{-5t}"
- Hint 3: "After multiplying: (e^{-5t}·x)' = 2t + 2B"
- Hint 4: "Integrate: e^{-5t}·x = t² + 2Bt + A. Isolate x."

## If Step 3 (solving z):
- Hint 1: "Substitute y into the third equation. You get: z' - 5z = 3y = 3(t+B)e^{5t}"
- Hint 2: "Again, the integrating factor is μ = e^{-5t}"
- Hint 3: "After multiplying: (e^{-5t}·z)' = 3t + 3B"
- Hint 4: "Integrate: e^{-5t}·z = (3/2)t² + 3Bt + C. Isolate z."

# COMMON ERRORS TO CHECK:
- Missing the constant of integration (B, A, or C)
- Wrong coefficient (should be 2 for x, 3 for z)
- Missing e^{5t} factor
- Wrong sign
- Forgot to substitute y correctly

# YOUR RESPONSE:
1. If CORRECT: "Correct! [brief confirmation]" and encourage next step
   - Accept any mathematically equivalent form of the correct answer
2. If INCORRECT: Identify the specific error and give the appropriate hint from above
3. If student asks for help/hint: Give the next hint in progression
4. After 3+ attempts: Give more explicit guidance, show intermediate steps
`;

  // Call the appropriate AI provider
  let hint;
  if (aiProvider === 'openrouter') {
    hint = await callOpenRouter(prompt);
  } else {
    // Default to Google Gemini
    hint = await callGemini(prompt);
  }

  return { hint, provider: aiProvider };
}

// Create server
const server = http.createServer(async (req, res) => {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  // API endpoint
  if (req.url === '/api/ai-hint' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', async () => {
      try {
        const data = JSON.parse(body);
        const result = await handleAIRequest(data);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (error) {
        console.error('AI API Error:', error);
        const aiProvider = process.env.AI_PROVIDER || 'google';
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          error: 'Error processing the request. Please try again.',
          provider: aiProvider,
          details: error.message
        }));
      }
    });
    return;
  }

  // Static files
  let filePath = req.url === '/' ? '/index.html' : req.url;
  filePath = path.join(__dirname, filePath);

  const ext = path.extname(filePath);
  const contentType = mimeTypes[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        res.writeHead(404);
        res.end('File not found');
      } else {
        res.writeHead(500);
        res.end('Server error');
      }
    } else {
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content);
    }
  });
});

server.listen(PORT, () => {
  const aiProvider = process.env.AI_PROVIDER || 'google';
  const providerName = aiProvider === 'openrouter' ? 'OpenRouter GPT-4o-mini' : 'Google Gemini 2.5 Flash';

  console.log(`\n🚀 Server running at http://localhost:${PORT}`);
  console.log(`\n🤖 AI Provider: ${providerName}`);
  console.log(`\n📝 Open this URL in your browser to test the exercise`);
  console.log(`\n🧐 Click "Digital Friend" to test the AI assistant`);
  console.log(`\n   Press Ctrl+C to stop the server\n`);
});
