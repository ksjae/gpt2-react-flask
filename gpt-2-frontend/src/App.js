import React, { useState } from 'react';
import Button from './components/Button';
import TextBox from './components/TextBox';
import './styles.scss';
import { postGenerateTextEndpoint } from './utils';

function App() {
  const [text, setText] = useState("");
  const [model, setModel] = useState('gpt2');
  const [generatedText, postGenerateText] = postGenerateTextEndpoint();

  const generateText = () => {
    postGenerateText({ text, model, userId: 1 });
  }

  return (
    <div className='app-container'>
      <form noValidate autoComplete='off'>
        <h1>아무말 대잔치</h1>
        <h3>독서록, 자소서, 레포트 등에 들어갈 내용을 인공지능이 만들어냅니다.</h3>
        <TextBox text={text} setText={setText} />
        <Button onClick={generateText} />
        <br />
        <a href="https://github.com/ksjae/KoGPT2-large">ksjae 제작</a>
      </form>

      {generatedText.pending &&
        <div className='result pending'>잠깐만 기다려 주세요...</div>}

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
