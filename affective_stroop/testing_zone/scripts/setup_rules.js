const fs = require("fs");

// Define the new content for the firestore.rules file
const newRulesContent = `rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /{document=**} {
      allow read, write: if false;
    }
    match /autora/autora_in/conditions/{id} {
      allow read, write: if true
    }
    match /autora/autora_meta {
      allow read, write: if true
    }
    match /autora/autora_out/observations/{id} {
     allow write: if true
    }
    match /autora/autora_out/data_all/{id} {
     allow write: if true
    }
   }
}`;

// Write the new content to the firestore.rules file
fs.writeFileSync("./firestore.rules", newRulesContent);
