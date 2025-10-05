import React from 'react';
import ReactDOM from 'react-dom/client';
import Error from './Error';
import {waitPage, endPage} from "./pages";
import {getCondition, setObservation, setBackup} from "autora-firebase-functions";
import main from "./design/main"
import db from "./firebase"

const root = ReactDOM.createRoot(document.getElementById('root'));

const index = async () => {
    if (process.env.NODE_ENV === 'development' && process.env.REACT_APP_devNoDb === 'True') {
        await main(0, 0)
        return
    }
    let prolificId = null
    if (process.env.REACT_APP_useProlificId === 'True') {
        const queryString = window.location.search;
        const urlParams = new URLSearchParams(queryString);
        prolificId = urlParams.get('PROLIFIC_PID');
    }
    let condition = await getCondition(db, 'autora', prolificId)
    if (condition && (prolificId !== null || process.env.REACT_APP_useProlificId === 'False')) {
        const observation = await main(condition[0], condition[1])
        waitPage()
        await setObservation(db, 'autora', condition[0], observation)
        await setBackup(db, 'autora', condition[0], condition[1], observation)
        endPage()
    } else {
        root.render(
            <React.StrictMode>
                <Error/>
            </React.StrictMode>
        );
    }
}
await index()