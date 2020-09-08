import { FormControl, InputLabel, MenuItem, Select } from '@material-ui/core';
import React from 'react';


const SelectBox = ({ length, setLength }) => {
    const inputLabel = React.useRef(null);
    const [labelWidth, setLabelWidth] = React.useState(0);
    React.useEffect(() => {
        setLabelWidth(inputLabel.current.offsetWidth);
    }, []);

    return (
        <FormControl
            margin='normal'
            style={{ width: '200px' }}
            variant='outlined'>
            <InputLabel ref={inputLabel} htmlFor="model-select">길이</InputLabel>
            <Select
                value={length}
                labelWidth={labelWidth}
                onChange={e => setLength(e.target.value)}
                inputProps={{
                    name: 'length',
                    id: 'model-select',
                }}
            >
                <MenuItem value={50}>50단어</MenuItem>
                <MenuItem value={100}>100단어</MenuItem>
                <MenuItem value={150}>150단어</MenuItem>
            </Select>
        </FormControl>
    )
};

export default SelectBox;