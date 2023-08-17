import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function analyzeData(filePath) {
    const rawData = await fs.promises.readFile(filePath, 'utf-8');
    const data = JSON.parse(rawData);

    const counts = {};

    for (const key in data[0]) {
        if (key !== 'date') {
            counts[key] = {};
        }
    }

    for (const entry of data) {
        for (const key in counts) {
            const value = entry[key];
            if (!counts[key][value]) {
                counts[key][value] = { frequency: 0, dates: [] };
            }
            counts[key][value].frequency++;
            counts[key][value].dates.push(entry.date);
        }
    }

    const mostFrequent = {};
    const totalEntries = data.length;

    for (const key in counts) {
        mostFrequent[key] = Object.entries(counts[key])
            .reduce((a, b) => a[1].frequency > b[1].frequency ? a : b);
        mostFrequent[key] = {
            number: parseInt(mostFrequent[key][0], 10),
            frequency: mostFrequent[key][1].frequency,
            probability: (mostFrequent[key][1].frequency / totalEntries) * 100,
            dates: mostFrequent[key][1].dates
        };
    }

    return { mostFrequent };
}


const filePath = join(__dirname, 'lotto.json');
analyzeData(filePath).then(({ mostFrequent }) => {
    for (const key in mostFrequent) {
        console.log(`For ${key}:`);
        console.log(`Most frequent number: ${mostFrequent[key].number}`);
        console.log(`Frequency: ${mostFrequent[key].frequency}`);
        console.log(`Probability: ${mostFrequent[key].probability.toFixed(2)}%`);
        console.log('----------------------------------');
    }
});