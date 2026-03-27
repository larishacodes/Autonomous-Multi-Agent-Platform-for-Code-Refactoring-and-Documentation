const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();

app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });

/**
 * 🔥 Python Pipeline Caller
 */
function callPythonPipeline(code, option) {
    return new Promise((resolve, reject) => {
        const path = require('path');

        const py = spawn('python', [
            path.join(__dirname, '..', '..', 'api.py')
        ]);
        let output = '';
        let error = '';

        py.stdin.write(JSON.stringify({
            code: code,
            mode: option
        }));
        py.stdin.end();

        py.stdout.on('data', (data) => {
            output += data.toString();
        });

        py.stderr.on('data', (data) => {
            error += data.toString();
        });

        py.on('close', () => {
            if (error) {
                reject(error);
            } else {
                try {
                    resolve(JSON.parse(output));
                } catch (e) {
                    reject("Invalid JSON from Python: " + output);
                }
            }
        });
    });
}

/**
 * 🔥 MAIN ROUTE
 */
app.post('/api/process-file', upload.single('file'), async (req, res) => {
    try {
        const file = req.file;
        const option = req.body.option;

        if (!file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        const originalContent = file.buffer.toString('utf-8');

        console.log("🚀 Sending to Python pipeline...");

        const pythonResponse = await callPythonPipeline(originalContent, option);

        if (!pythonResponse.success) {
            return res.status(500).json({ error: pythonResponse.error });
        }

        const result = pythonResponse.result;

        res.json({
            originalContent: originalContent,
            processedContent: result.refactored_code || result.documentation || "",
            refactoredContent: result.refactored_code,
            documentedContent: result.documentation,
            option: option,
            summary: "Processed via multi-agent pipeline",
            stats: result.refactor_evaluation || {}
        });

    } catch (err) {
        console.error("❌ ERROR:", err);
        res.status(500).json({ error: "Pipeline failed", details: err.toString() });
    }
});

app.listen(3001, () => {
    console.log('✅ Server running on port 3001');
});