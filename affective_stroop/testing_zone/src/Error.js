/**
 * This is the error page that is shown if there are no conditions in the database
 */
function Error() {
    return (
        <div className="App">
            <header className="App-header">
                <p>
                    We are sorry, there has been an unexpected technical issue.<br/>
                    Thank you for your understanding.
                </p>

                <a
                    className="App-link"
                    href="https://app.prolific.co"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    Prolific
                </a>
            </header>
        </div>
    );
}

export default Error;
