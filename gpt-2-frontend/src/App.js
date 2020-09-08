import React, { useState } from 'react';
import Button from './components/Button';
import TextBox from './components/TextBox';
import SelectBox from './components/SelectBox';
import './styles.scss';
import { postGenerateTextEndpoint } from './utils';

function App() {
    const [text, setText] = useState("");
    const [length, setLength] = useState(50);
    const [generatedText, postGenerateText] = postGenerateTextEndpoint();

    const generateText = () => {
        postGenerateText({ text, length, userId: 1 });
    }

    return (
        <div className='app-container'>
            <form noValidate autoComplete='off'>
                <h1>아무말 대잔치</h1>
                <h3>독서록, 자소서, 레포트 등에 들어갈 내용을 인공지능이 만들어냅니다.</h3>
                <TextBox text={text} setText={setText} />
                <SelectBox text={length} setLength={setLength} />
                <Button onClick={generateText} />
                <br />
                <a href="https://github.com/ksjae/KoGPT2-large">ksjae 제작</a>
                <br />
                <a href="http://aihub.or.kr/">한국정보화진흥원 고성능 컴퓨팅 지원 사업의 도움을 받음</a>
                <p>본 프로그램의 이용 내역은 연구를 위해 활용됩니다.</p>
            </form>

            {generatedText.pending &&
                <div className='result pending'>잠시(1~5분) 기다려 주세요...</div>}

            {generatedText.complete &&
                (generatedText.error ?
                    <div className='result error'>Bad Request</div> :
                    <div className='result valid'>
                        {generatedText.data.result}
                    </div>)}
        </div>
    );
}

export default App;
