import fs from 'fs/promises';

function parseData(inputString) {
    // Split the string by new line to get each row
    const rows = inputString.trim().split('\n');

    // Remove the header row
    const header = rows.shift().split('\t');

    // Parse each row
    const results = rows.map(row => {
        const values = row.split('\t');
        const obj = {};
        for (let i = 0; i < header.length; i++) {
            if (values[i] !== undefined && values[i] !== "") {
                if (["Tag", "Monat", "Jahr", "Zahl1", "Zahl2", "Zahl3", "Zahl4", "Zahl5", "Zahl6", "Zusatzzahl", "Superzahl"].includes(header[i])) {
                    obj[header[i]] = parseInt(values[i]);
                } else {
                    obj[header[i]] = values[i];
                }
            }
        }
        obj["date"] = `${values[header.indexOf("Jahr")]}-${String(values[header.indexOf("Monat")]).padStart(2, '0')}-${String(values[header.indexOf("Tag")]).padStart(2, '0')}`;
        delete obj["Tag"];
        delete obj["Monat"];
        delete obj["Jahr"];
        return obj;
    });

    return results;
}

async function main() {
    try {
        const data = await fs.readFile('./data/lotto.csv', 'utf-8');
        const jsonData = parseData(data);
        console.log(jsonData);
        await fs.writeFile('./data/lotto_parsed.json', JSON.stringify(jsonData, null, 2));
    } catch (error) {
        console.error('Error:', error);
    }
}

main();