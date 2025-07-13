# User Testing Guide for RAG-Enhanced ChatGPT

## Overview
This guide provides step-by-step instructions for manually testing the RAG-Enhanced ChatGPT application to ensure all features work correctly.

## Prerequisites
- Application is running (see `how_to_run.md`)
- OpenAI and Milvus services are configured
- Web browser for UI testing
- Terminal/Command prompt for API testing

## Test Scenarios

### 1. System Health Check

#### Objective
Verify that all system components are properly connected and functioning.

#### Steps
1. **Web Interface Access**:
   - Open browser and navigate to `http://localhost:8000`
   - Verify the main page loads with the chat interface
   - Check that settings panel is visible on the right side

2. **API Health Check**:
   ```bash
   curl http://localhost:8000/health/status
   ```
   
3. **Expected Results**:
   - Web page loads without errors
   - Settings panel shows green status indicators
   - API returns:
   ```json
   {
     "milvus_connected": true,
     "collection_exists": true,
     "total_documents": 0,
     "openai_connected": true
   }
   ```

#### Troubleshooting
- Red status indicators suggest service connection issues
- Check environment variables and service credentials

---

### 2. Document Upload Testing

#### Test 2.1: Text Upload via Web Interface

**Objective**: Test document upload through the web UI

**Steps**:
1. Click "Upload Documents" button in the web interface
2. Select "Text Input" tab
3. Enter the following test data:
   - **Title**: "Machine Learning Basics"
   - **Content**: 
   ```
   Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to perform tasks without explicit instructions. It relies on patterns and inference instead. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.
   ```
4. Click "Upload Text"

**Expected Results**:
- Success message appears
- Document count in settings panel increases
- Upload modal closes automatically

**API Equivalent**:
```bash
curl -X POST "http://localhost:8000/documents/upload-text" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Machine Learning Basics",
    "text": "Machine learning is a subset of artificial intelligence..."
  }'
```

#### Test 2.2: File Upload via Web Interface

**Objective**: Test file upload functionality

**Steps**:
1. Create a test file `test_document.txt` with content:
   ```
   Natural Language Processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and humans using natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human languages in a manner that is valuable.
   ```
2. Click "Upload Documents" button
3. Select "File Upload" tab
4. Choose the `test_document.txt` file
5. Click "Upload Files"

**Expected Results**:
- File uploads successfully
- Document count increases
- File appears in document list

#### Test 2.3: Document List Verification

**Objective**: Verify uploaded documents appear in the system

**Steps**:
1. In the settings panel, expand "RAG Sources" section
2. Check that uploaded documents are listed
3. Verify document metadata (title, chunk count)

**API Equivalent**:
```bash
curl http://localhost:8000/documents/list
```

**Expected Results**:
- Documents appear with correct titles
- Chunk counts are greater than 0
- Document previews are shown

---

### 3. Chat Functionality Testing

#### Test 3.1: Basic Chat Without RAG

**Objective**: Test basic chat functionality without document context

**Steps**:
1. In the chat interface, type: "What is the capital of France?"
2. Press Enter or click Send
3. Wait for response

**Expected Results**:
- Response appears within 5-10 seconds
- Answer is correct (Paris)
- No RAG sources shown (empty sources section)

**API Equivalent**:
```bash
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?"}'
```

#### Test 3.2: RAG-Enhanced Chat

**Objective**: Test chat with document context retrieval

**Prerequisites**: Complete document upload tests first

**Steps**:
1. Type: "What is machine learning according to the uploaded documents?"
2. Send the message
3. Wait for response

**Expected Results**:
- Response incorporates information from uploaded documents
- RAG sources section shows relevant document chunks
- Response mentions document-specific content
- Source documents are highlighted with similarity scores

#### Test 3.3: Follow-up Questions

**Objective**: Test conversation continuity

**Steps**:
1. Ask: "Can you explain more about NLP?"
2. Then ask: "How does it relate to the previous topic?"

**Expected Results**:
- Both questions receive relevant answers
- Second question shows understanding of context
- Appropriate RAG sources are retrieved for each question

---

### 4. Error Handling Testing

#### Test 4.1: Invalid Input Handling

**Objective**: Test system behavior with invalid inputs

**Steps**:
1. **Empty Message Test**:
   - Try sending an empty message
   - Expected: Error message or prevention of sending

2. **Very Long Message Test**:
   - Send a message with 2000+ characters
   - Expected: Message handled gracefully or appropriate error

3. **Invalid Document Upload**:
   - Try uploading an empty text document
   - Expected: Appropriate error message

#### Test 4.2: Service Failure Simulation

**Objective**: Test error handling when services are unavailable

**Note**: This requires temporarily modifying configuration

**Steps**:
1. Stop the application
2. Temporarily change OpenAI API key to invalid value in `.env`
3. Restart application
4. Try to send a chat message

**Expected Results**:
- Appropriate error message displayed
- System doesn't crash
- User-friendly error explanation

---

### 5. Performance Testing

#### Test 5.1: Large Document Upload

**Objective**: Test system with larger documents

**Steps**:
1. Create a document with 5000+ words
2. Upload via text input
3. Monitor processing time and memory usage

**Expected Results**:
- Document processes within reasonable time (< 2 minutes)
- Multiple chunks created
- System remains responsive

#### Test 5.2: Multiple Concurrent Uploads

**Objective**: Test concurrent document processing

**Steps**:
1. Open multiple browser tabs
2. Upload different documents simultaneously
3. Monitor system behavior

**Expected Results**:
- All uploads complete successfully
- No data corruption or mixing
- System remains stable

#### Test 5.3: Rapid Chat Messages

**Objective**: Test chat responsiveness under load

**Steps**:
1. Send multiple questions quickly (one every 2-3 seconds)
2. Monitor response times and quality

**Expected Results**:
- All messages receive responses
- Response quality remains consistent
- No significant delays or timeouts

---

### 6. Data Persistence Testing

#### Test 6.1: Application Restart

**Objective**: Verify data persists after restart

**Steps**:
1. Upload several documents
2. Note document count
3. Stop and restart the application
4. Check document count and list

**Expected Results**:
- Document count matches before restart
- All uploaded documents still accessible
- Chat functionality works with existing documents

---

### 7. Browser Compatibility Testing

#### Test 7.1: Cross-Browser Testing

**Objective**: Ensure web interface works across browsers

**Browsers to Test**:
- Chrome/Chromium
- Firefox
- Safari (if on macOS)
- Edge (if on Windows)

**Steps**:
1. Open application in each browser
2. Test document upload
3. Test chat functionality
4. Verify UI layout and responsiveness

**Expected Results**:
- Consistent functionality across browsers
- Proper UI rendering
- No console errors

---

### 8. Mobile Responsiveness Testing

#### Test 8.1: Mobile Interface

**Objective**: Test mobile device compatibility

**Steps**:
1. Open application on mobile device or use browser developer tools mobile view
2. Test all core functionality:
   - Document upload
   - Chat interface
   - Settings panel

**Expected Results**:
- Interface adapts to mobile screen
- All buttons and inputs are accessible
- Text is readable without zooming

---

## Test Results Documentation

### Test Results Template

For each test scenario, document:

```
Test: [Test Name]
Date: [Date]
Tester: [Name]
Status: PASS/FAIL/PARTIAL
Notes: [Any observations or issues]
Performance: [Response times, if applicable]
```

### Common Issues and Solutions

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Document upload fails | Large file size | Check file size limits |
| Chat responses are slow | Network latency | Check internet connection |
| RAG sources not shown | No relevant documents | Upload more diverse content |
| UI not loading | JavaScript errors | Check browser console |
| API errors | Service configuration | Verify environment variables |

### Performance Benchmarks

Document these metrics during testing:

- **Document Upload Time**: < 30 seconds for 1000 words
- **Chat Response Time**: < 10 seconds for simple queries
- **RAG Response Time**: < 15 seconds with document retrieval
- **System Startup Time**: < 30 seconds
- **Memory Usage**: Monitor for memory leaks during extended use

### Success Criteria

The application passes testing if:

✅ All core functionality works as expected  
✅ Error handling is graceful and informative  
✅ Performance meets acceptable benchmarks  
✅ Data persistence works correctly  
✅ UI is responsive and accessible  
✅ No critical bugs or crashes occur  

### Reporting Issues

When reporting issues, include:

1. **Steps to reproduce**
2. **Expected vs. actual behavior**
3. **Environment details** (OS, browser, Python version)
4. **Error messages** (from UI and logs)
5. **Screenshots** (if UI-related)

Check `logs/app.log` for detailed error information when issues occur.