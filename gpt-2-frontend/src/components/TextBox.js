import React from 'react';
import { TextField } from '@material-ui/core';

const TextBox = ({ text, setText }) => (
    <TextField
        margin='normal'
        label="인공지능이 여기 있는 말 뒤의 내용을 생성합니다."
        variant="outlined"
        fullWidth
        multiline
        rows='4'
        value={text}
        onChange={e => setText(e.target.value)}
    />
);

export default TextBox;
