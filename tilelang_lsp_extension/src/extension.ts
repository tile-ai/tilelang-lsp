//npm run compile
//vsce package
import { match } from 'assert';
import * as vscode from 'vscode';

// global decorator
let typeHintDecoration: vscode.TextEditorDecorationType | undefined;

export function activate(context: vscode.ExtensionContext) {
    
    const disposable = vscode.commands.registerCommand('my-python-type-inspector.showHints', () => {
        showTypeHints();
    });
    context.subscriptions.push(disposable);

    vscode.window.onDidChangeActiveTextEditor(editor => {
        if (editor?.document.languageId === 'python') {
            showTypeHints();
        }
    });

    vscode.workspace.onDidChangeTextDocument(event => {
        if (event.document.languageId === 'python' && 
            vscode.window.activeTextEditor?.document === event.document) {
            // wait...
            setTimeout(showTypeHints, 300);
        }
    });

    // check for the initialization
    if (vscode.window.activeTextEditor?.document.languageId === 'python') {
        showTypeHints();
    }
}

// check if the input line is a function parameter....
// the returned value is the line number of the function...
// If the expression is not in the parameter table, then return 0....
function IsFuncParam(lines:string[],line_num_input:number):number{
    
    let line_num = line_num_input;
    const trimmed_input = lines[line_num].trim();
    const space=GetLeadingSpaceLength(lines[line_num]);
    let max_space = space;
    
    for(let i = line_num_input;i>=0;i--){
        const trimmed = lines[i].trim();
        const space_prob=GetLeadingSpaceLength(trimmed);
        if (trimmed.startsWith('#') || trimmed === '') continue;

        const pattern = new RegExp(/^def\.*/);
        const pattern2 = new RegExp(/^\)\:\.*/);
        const customMatch = trimmed.match(pattern);
        const customMatch2 = trimmed.match(pattern2);
        if(customMatch && max_space>space_prob){
            
            line_num=i;
            return line_num;
        }
        if(customMatch2){
            line_num=0;
            return line_num;
        }
        
        //update maximum space
        if(max_space<space_prob){ max_space=space_prob;}

    }
    return 0;
}

function GetModuleImportName(lines:string[],module_name:string):{exist:boolean,name:string}{
    module_name = module_name.trim();
    const pattern_import = new RegExp(`^import\\s*${module_name}\\s*$`);
    const pattern_import_as =new RegExp(`^import\\s*${module_name}\\s*as\\s*([^#]*)$`);
    const pattern_from_import_as =new RegExp(`^from\\s*.*\\s*import\\s*${module_name}\\s*as\\s*([^#]*)$`);
    for(let i=0;i<lines.length;i++){
        const trimmed=lines[i].trim();
        if (trimmed.startsWith('#') || trimmed === '') continue;
        const match_from_import_as=trimmed.match(pattern_from_import_as);
        if(match_from_import_as){
            return {exist:true,name: match_from_import_as[1].trim()};
        }
        const match_import_as=trimmed.match(pattern_import_as);
        if(match_import_as){
            return {exist:true,name: match_import_as[1].trim()};
        }
        const match_import =trimmed.match(pattern_import);
        if(match_import){
            return {exist:true,name:module_name};
        }

    }
    return {exist:false,name:module_name};
}



//get the leading space number
function GetLeadingSpaceLength(str:string) {
  return str.length - str.trimStart().length;
}

//find all
function FindAllOccurrencesOfVariable(
    document: vscode.TextDocument,
    varName: string,var_input_linenum:number,lines:string[]
): vscode.Position[] {
    const text = document.getText();
    const positions: vscode.Position[] = [];
    
    // \b make sure find a whole word...
    const regex = new RegExp(`\\b${varName}\\b`, 'g');
    let match: RegExpExecArray | null;
    const func_region_input=GetFuncRegion(lines,var_input_linenum);
    

    while ((match = regex.exec(text)) !== null) {
        const offset = match.index;
        const position = document.positionAt(offset);
        const line_num=position.line;
        //console.log(line_num);
        const func_region_probe=GetFuncRegion(lines,line_num);
        const trimmed = lines[line_num].trim();
        if (trimmed.startsWith('#') || trimmed === '') continue;
        
        //if both of them are in the same function
        //func_region_input==func_region_probe && var_input_linenum<line_num
        //let flag = func_region_input.start==func_region_probe.start;

        const a = func_region_input.start;
        const b = func_region_probe.start
        if(a==b && var_input_linenum<=line_num){

            //if there is another defination, stop and return....
            const pattern_re_define = new RegExp(`^\\s*${varName}\\s*=\\s*[^=].*$`);
            const match = trimmed.match(pattern_re_define);
            if(match && var_input_linenum<line_num){
                
                return positions;
            }
            
            // otherwise, decorate another one....
            positions.push(position);
            //decorate "x" in "x = [type:fragment] "
            const pattern = new RegExp(`^([a-zA-Z_][\\w\\.]*)\\s*=\\s*${varName}\\s*$`);
            const find = findOneOccurencesbyRegex(document,varName,position.line,pattern,positions);
            
            if(find.exist){
                positions.push(new vscode.Position(find.row,find.col));
            }
        }



    }

    return positions;
}


//find all with regex input from some line...
function findOneOccurencesbyRegex(document: vscode.TextDocument,
    varName: string, linestart:number,pattern:RegExp,p_input:vscode.Position[]):{row:number,col:number, exist:Boolean}{
        
    let flag = false;
    let varStartIndex = 0;
    const text = document.getText();
    const lines = text.split('\n');
    let positions: vscode.Position[] = [];
    const lineIndex = linestart;
    const match = lines[lineIndex].trim().match(pattern);
    
    if (match) {
        //console.log("Hello, world!"); // 
        const leftVar = match[1]
        const originalLine = lines[lineIndex];
        varStartIndex = originalLine.indexOf(leftVar);

        if (varStartIndex !== -1) {
            const position_temp = new vscode.Position(lineIndex, varStartIndex);
            flag = true;
        }
    }

    if(!flag){
        
        return { row:0,col:0 , exist: false};
    }
    else{
       
        return {row:lineIndex,col:varStartIndex , exist:true};

    }

}

function FindDefinationbyRegex(decorations:vscode.DecorationOptions[], document:vscode.TextDocument,trimmed:string,
    lineIndex:number,lines:string[],pattern:RegExp, dec_string:string):vscode.DecorationOptions[]{
        const line = lines[lineIndex];
        const editor = vscode.window.activeTextEditor;
        const customMatch = trimmed.match(pattern);
        // The regex pattern has to follw:
        // match(/^([a-zA-Z_]\w*)\s*=\s*([a-zA-Z_]\w*)\.alloc_fragment\s*\([^,)]+(?:,[^,)]+)*\)/)
        // define fragment e.g. s_acc=T.alloc_fragment()...
        if (customMatch) {
            const varName = customMatch[1];
            const className = customMatch[2]; // 
            //check if the alloc function is imported from the tilelang module...
            let import_name=GetModuleImportName(lines,"tilelang.language");
            const exist=import_name.exist;
            if (!exist || customMatch[2].trim()!= import_name.name){return decorations;}
            // findAllOcc implies considering the function regions....         
            const allBufPositions = FindAllOccurrencesOfVariable(document, varName,lineIndex,lines);
            const typeHint0=dec_string;
            let type_with_dim: string =typeHint0;
            //type_with_dim = type_with_dim.concat("[",customMatch[3],"]");
            //type_with_dim = type_with_dim.concat("[",FindInterpretation(customMatch[3],lines,lineIndex,GetLeadingSpaceLength(line)),"]");
            type_with_dim = type_with_dim.concat("[",TilelangParamInterp(customMatch[3],lines,lineIndex),"]");
            const typeHint1 = type_with_dim.replace(/\s/g, '');

                // find position and create decoration
                // process
            for (let i = 1; i < allBufPositions.length; i++) {
                const pos = allBufPositions[i];
                    // hint or decoration
                const insertPos = new vscode.Position(
                    pos.line,
                    pos.character + varName.length
                );

                decorations.push({
                    range: new vscode.Range(insertPos, insertPos), // zero width...not quite clear about it...
                    renderOptions: {
                        before: {
                            contentText: typeHint1
                        }
                    }
                });

                    
            }

            const typeMap: Record<string, string> = {
                'T': 'BufferType',
                'Array': 'ArrayType',
                'Tensor': 'Tensor'
            // more mappings are possible...
            };
    
            //const typeHint = ': ' + (typeMap[className] || className);
            const typeHint = dec_string;
            
            const varStart = line.indexOf(varName);
            if (varStart === -1) return decorations;
            
            const startPos = new vscode.Position(lineIndex, varStart + varName.length);
            decorations.push({
                range: new vscode.Range(startPos, startPos),
                renderOptions: {
                    before: { contentText: typeHint1 }
                }
            });
            
        }

    return decorations;

}


function GetValidRegionToBottom(document:vscode.TextDocument,lines:string[],varName:string,line_num:number):{startline:number,lastline:number}{
    //get the initial space of the line with index [line_num] ....
    let startline = line_num;
    let lastline = line_num;
    const lastlinedoc=lines.length;
    let space_num = GetLeadingSpaceLength(lines[line_num]);
    
    for(let i = startline; i < lastlinedoc;i++){
        const line = lines[i];
        const trimmed = line.trim();
        // skip // and empty lines...
        if (trimmed.startsWith('#') || trimmed === '') continue;

        const space_num_prob=GetLeadingSpaceLength(line);
        if (space_num_prob<space_num){
            lastline = i;
            return {startline:startline,lastline:lastline};
        }
    }

    return {startline:startline,lastline:lastlinedoc};

}

function IsBracketError(str:string):boolean{
    let bracketLevel = 0;
    let error = false;

    for (let i = 0; i < str.length; i++) {
        const char = str[i];

        if (char === '[') {
        bracketLevel++;
        } else if (char === ']') {
        bracketLevel--;
        } 
        if(bracketLevel<0){
            return true;
        }

    }
    if(bracketLevel != 0){
        return true;
    }
    return false;

}
function splitPreservingBrackets(str:string):string[] {
  const result:string[] = [];
  let current = '';
  let bracketLevel = 0;

  for (let i = 0; i < str.length; i++) {
    const char = str[i];

    if (char === '[') {
      bracketLevel++;
      current += char;
    } else if (char === ']') {
      bracketLevel--;
      current += char;
    } else if (char === ',' && bracketLevel === 0) {
      
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }

  if (current !== '') {
    result.push(current.trim());
  }

  return result;
}
function TilelangParamInterp(input_str:string,lines:string[],line_num:number):string{
    
    // Not quite sure if we need to erase spaces yet....
    const input_no_space = input_str.replace(/\s/g, '');
    let input = input_no_space.replace(/\(/g, '[');
    input = input.replace(/\)/g, ']');
    
    if(IsBracketError(input_str)) {return "*[Bracket Error!!!]";}
    const words=splitPreservingBrackets(input);
    let res:string = '';
    const line = lines[line_num];
    for(let i =0; i < words.length ; i++){
        const word=words[i];
        //match the "[" and "]"...
        const pattern:RegExp = /^\s*\[\s*(.*)\s*\]/ ;
        const match=word.match(pattern);
        
        if(match){
            
            res = res.concat("[",TilelangParamInterp(match[1],lines,line_num),"]");
            if(i<words.length-1) {res = res.concat(",")}

        }
        else{
            res = res.concat(FindInterpretation(word,lines,line_num,GetLeadingSpaceLength(line)));
            
            if(i<words.length-1) {res = res.concat(",")}
        }

    }
    return res;
}

function FindInterpretation(varName:string,lines:string[],line_num:number,space_pos:number):string{
    let res = varName;
    const space_num = GetLeadingSpaceLength(lines[line_num]);
    const func_region_input=GetFuncRegion(lines,line_num);
    const func_pos_input = func_region_input.start;
    
    console.log(line_num,varName,func_region_input);

    //If the phrase to be interpretated is the input parameter of a function....
    
    if(IsFuncParam(lines,line_num)){
        if (IsFuncParam(lines,line_num)==line_num){
            return FindInterpretation(varName,lines,IsFuncParam(lines,line_num)-1,GetLeadingSpaceLength(lines[IsFuncParam(lines,line_num)]));
        }
        return FindInterpretation(varName,lines,IsFuncParam(lines,line_num),GetLeadingSpaceLength(lines[IsFuncParam(lines,line_num)]));
    }



    // If the varName begins with "dtype =", erase it...
    const trimmed_var = varName.trim();
    const pattern_dype = new RegExp(/^\s*dtype\s*=(.*)/);
    const match_dtype = trimmed_var.match(pattern_dype);
    if (match_dtype){
        return FindInterpretation(match_dtype[1],lines,line_num,space_pos);
    }
    // If the varName begins with "scope =", erase it...
    const pattern_scope = new RegExp(/^\s*scope\s*=(.*)/);
    const match_scope = trimmed_var.match(pattern_scope);
    if (match_scope){
        return FindInterpretation(match_scope[1],lines,line_num,space_pos);
    }



    // find the line in form " [varName] = xxxxx " with a loop
    for(let i = line_num - 1; i >=func_pos_input ;i--){
        const line = lines[i];
        const trimmed = line.trim();
        if (trimmed.startsWith('#') || trimmed === '') continue;
        const func_region_probe = GetFuncRegion(lines,i);
        const is_param = IsFuncParam(lines,i);
        if (is_param){

            
            if (is_param<func_pos_input){break;}
            else {continue;}
        }


        //ignore expressions with shorter life time (Those expression will not affect the value of this line)...
        const line_space_num_prob = GetLeadingSpaceLength(line);

        //Now, check if we have " [varName] = xxxxx "
        
        const pattern = new RegExp(`^${varName}\\s*=\\s*(.*)\\s*$`);        
        const customMatch = trimmed.match(pattern);
        if(customMatch){
            const rightside = customMatch[1].trim();
            if (line_space_num_prob > space_num) return varName;
            //if it is a number, return...
            const match_num = rightside.match(/^([+-]?(?:\d+\.?\d*|\.\d+))$/);
            if (match_num){
                return match_num[1];
            }
            else{
                return FindInterpretation(rightside,lines,i,line_space_num_prob);
            }

        }
    }
    return res;
}

function GetFuncRegion(lines:string[],line_num:number):{start:number,last:number,func_space_num:number}{
    const line_space = GetLeadingSpaceLength(lines[line_num]);
    let function_start=line_num;
    let func_space_num=line_space;
    //get the func start line number [function_start], and the lateral positon of the function [func_space_num]
    // the loop changed from i=line_num to i = line_num-1....
    for (let i =line_num;i>=0;i--){
        const line = lines[i];
        const trimmed=line.trim();
        // skip // and empty lines...
        if (trimmed.startsWith('#') || trimmed === '') continue;
        const space_num_prob=GetLeadingSpaceLength(lines[i]);
        //strictly left...
        if(space_num_prob<func_space_num){
            
            //Now, check if we have " def (.... "
            const pattern = new RegExp(/^def\s.*/);
            const pattern_class =new RegExp(/^class\s.*/)
            const pattern2 = new RegExp(/^\)\:.*/);
            const customMatch = trimmed.match(pattern);
            const customMatch_class = trimmed.match(pattern_class);
            if(trimmed.match(pattern2)) {continue;}
            func_space_num = space_num_prob;
            if(customMatch || customMatch_class){
                function_start = i;
                func_space_num = space_num_prob;
                break;
            }

        }
        if(i==0){function_start = 0; func_space_num=-1;}
    }
    let function_end = line_num;
    //get the func end line number
    for(let i =line_num+1;i<lines.length;i++){
        const line = lines[i];
        const trimmed=line.trim();
        // skip // and empty lines...
        if (trimmed.startsWith('#') || trimmed === '') continue;
        if (trimmed.startsWith(')') || trimmed === '') continue;
        const space_num_prob=GetLeadingSpaceLength(lines[i]);
        function_end = i-1;
        if (space_num_prob<=func_space_num){
            return {start:function_start,last:function_end,func_space_num:func_space_num}
        }

    }
    return {start:function_start,last:lines.length,func_space_num:func_space_num};
}


function showTypeHints() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'python') return;
    
    // destroy the old decoration
    if (typeHintDecoration) {
        typeHintDecoration.dispose();
    }

    // create a new decoration
    typeHintDecoration = vscode.window.createTextEditorDecorationType({
        before: {
            color: new vscode.ThemeColor('editorInlayHint.foreground'),
            backgroundColor: new vscode.ThemeColor('editorInlayHint.background'),
            fontStyle: 'italic',
            margin: '0 0.5em 0 0',
            textDecoration: 'none; opacity: 0.7'
        }
    });

    let decorations: vscode.DecorationOptions[] = [];
    const text = editor.document.getText();
    const lines = text.split('\n');

    for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
        const line = lines[lineIndex];
        const trimmed = line.trim();
        
        // skip // and empty lines...
        if (trimmed.startsWith('#') || trimmed === '') continue;

        // the large function.... 20251224..... important.....
        // find the fragments.....
        //const pattern_fragment_defined:RegExp = /^([a-zA-Z_\.][\w.]*)\s*=\s*([a-zA-Z_][\w.]*)\.alloc_fragment\s*\(([^)]*)\)/ ;
        const pattern_fragment_defined:RegExp = /^([a-zA-Z_\.][\w.]*)\s*=\s*([a-zA-Z_][\w.]*)\.alloc_fragment\s*\((.*)\)/ ;
        decorations = FindDefinationbyRegex(decorations, editor.document, trimmed, lineIndex, lines,pattern_fragment_defined,': Tilelang.fragment');

        //find the shared memory tensor.......
        //const pattern_shared_defined:RegExp = /^([a-zA-Z_\.][\w.]*)\s*=\s*([a-zA-Z_][\w.]*)\.alloc_shared\s*\(([^)]*)\)/ ;
        const pattern_shared_defined:RegExp = /^([a-zA-Z_\.][\w.]*)\s*=\s*([a-zA-Z_][\w.]*)\.alloc_shared\s*\((.*)\)/ ;
        decorations = FindDefinationbyRegex(decorations, editor.document, trimmed, lineIndex, lines,pattern_shared_defined,': Tilelang.shared');
        
        const pattern_var_defined:RegExp = /^([a-zA-Z_\.][\w.]*)\s*=\s*([a-zA-Z_][\w.]*)\.alloc_var\s*\((.*)\)/ ;
        decorations = FindDefinationbyRegex(decorations, editor.document, trimmed, lineIndex, lines,pattern_var_defined,': Tilelang.var');

        //TODO: add other alloc methods.....



        //find the global tensor, defined as input variable.......
        const pattern_buffer_as_input_defined:RegExp = /^([a-zA-Z_\.][\w.]*)\s*:\s*([a-zA-Z_][\w.]*)\.buffer\s*\((.*)\)/ ;
        decorations = FindDefinationbyRegex(decorations, editor.document, trimmed, lineIndex, lines,pattern_buffer_as_input_defined,': Tilelang.buffer');

        const pattern_tensor_as_input_defined:RegExp = /^([a-zA-Z_\.][\w.]*)\s*:\s*([a-zA-Z_][\w.]*)\.tensor\s*\((.*)\)/ ;
        decorations = FindDefinationbyRegex(decorations, editor.document, trimmed, lineIndex, lines,pattern_tensor_as_input_defined,': Tilelang.tensor');

        const pattern_Tensor_as_input_defined:RegExp = /^([a-zA-Z_\.][\w.]*)\s*:\s*([a-zA-Z_][\w.]*)\.Tensor\s*\((.*)\)/ ;
        decorations = FindDefinationbyRegex(decorations, editor.document, trimmed, lineIndex, lines,pattern_Tensor_as_input_defined,': Tilelang.tensor');


    // int bool str, etc...
    const basicMatch = trimmed.match(/^([a-zA-Z_][\w.]*)\s*=\s*(.+)$/);
    if (!basicMatch) continue;


        // var = value
        const match = trimmed.match(/^([a-zA-Z_][\w.]*)\s*=\s*(.+)$/);
        if (!match) continue;

        const varName = match[1];
        const value = match[2];

        //
        let typeHint = '';
        if (/^".*"$|'.*'$/.test(value)) {
            typeHint = ': str';
        } else if (/^\d+$/.test(value)) {
            typeHint = ': int';
        } else if (value === 'True' || value === 'False') {
            typeHint = ': bool';
        } else if (value.startsWith('[') && value.endsWith(']')) {
            typeHint = ': list';
        } else if (value.startsWith('{') && value.endsWith('}')) {
            typeHint = ': dict';
        } else {
            continue; // skip for others
        }

        // find the line position
        const varStart = line.indexOf(varName);
        if (varStart === -1) continue;

        const startPos = new vscode.Position(lineIndex, varStart + varName.length);
        const endPos = new vscode.Position(lineIndex, varStart + varName.length);

        decorations.push({
            range: new vscode.Range(startPos, endPos),
            renderOptions: {
                before: {
                    contentText: typeHint
                }
            }
        });
    }

    // set the decorations...
    editor.setDecorations(typeHintDecoration, decorations);
}

export function deactivate() {
    if (typeHintDecoration) {
        typeHintDecoration.dispose();
    }
}