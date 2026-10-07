"""Bounded Python-subset interpreter for feedback controllers.

Model text is parsed as AST and interpreted, NEVER passed to eval/exec. It has
no filesystem, import, simulator, process, or network primitive. A resource-
limited worker adds containment for expensive calculations and memory use.
"""
import ast
import math


class ProgramError(ValueError):
    pass


class Returned(Exception):
    def __init__(self,value):self.value=value


class BreakLoop(Exception):pass
class ContinueLoop(Exception):pass


class Program:
    def __init__(self,source):
        if not isinstance(source,str) or not 1<=len(source)<=16000:
            raise ProgramError('code must contain 1..16000 characters')
        try:tree=ast.parse(source)
        except SyntaxError as e:raise ProgramError(str(e)) from e
        self.functions={};self.globals={};self.depth=0;self.fuel=100000
        for node in tree.body:
            if isinstance(node,ast.Import) and len(node.names)==1 and node.names[0].name=='math' and node.names[0].asname is None:
                continue
            if isinstance(node,ast.FunctionDef):
                if node.decorator_list or node.args.defaults or node.args.kw_defaults or node.args.vararg or node.args.kwarg or node.args.kwonlyargs or node.args.posonlyargs:
                    raise ProgramError('Functions use plain positional parameters, without decorators or defaults')
                self.functions[node.name]=node
            elif isinstance(node,ast.Assign):self.statement(node,self.globals)
            else:raise ProgramError('Top level supports import math, constants, and function definitions')
        if 'control' not in self.functions or len(self.functions['control'].args.args)!=2:
            raise ProgramError('Define control(obs, memory) with exactly two arguments')

    def tick(self):
        self.fuel-=1
        if self.fuel<0:raise ProgramError('Program instruction budget exceeded; shorten loops')

    def bounded(self,value):
        if isinstance(value,(str,list,tuple,dict,range)) and len(value)>4096:
            raise ProgramError('Intermediate collection/string exceeds 4096 elements')
        if isinstance(value,int) and value.bit_length()>512:raise ProgramError('Integer is too large')
        return value

    def call_function(self,name,args):
        node=self.functions[name]
        if len(args)!=len(node.args.args):raise ProgramError('Wrong argument count for '+name)
        self.depth+=1
        if self.depth>16:raise ProgramError('Function recursion limit exceeded')
        local=dict(zip([a.arg for a in node.args.args],args))
        try:
            for stmt in node.body:self.statement(stmt,local)
        except Returned as r:return r.value
        finally:self.depth-=1

    def control(self,obs,memory):
        self.fuel=100000
        return self.call_function('control',[obs,memory])

    def binary(self,op,a,b):
        if isinstance(op,ast.Add):
            if isinstance(a,(str,list,tuple)) and isinstance(b,type(a)) and len(a)+len(b)>4096:raise ProgramError('Collection addition too large')
            return self.bounded(a+b)
        if isinstance(op,ast.Sub):return self.bounded(a-b)
        if isinstance(op,ast.Mult):
            for seq,count in ((a,b),(b,a)):
                if isinstance(seq,(str,list,tuple)) and isinstance(count,int) and len(seq)*max(0,count)>4096:raise ProgramError('Collection multiplication too large')
            return self.bounded(a*b)
        if isinstance(op,ast.Div):return a/b
        if isinstance(op,ast.FloorDiv):return self.bounded(a//b)
        if isinstance(op,ast.Mod):
            if isinstance(a,str):raise ProgramError('String interpolation via % is unsupported')
            return self.bounded(a%b)
        if isinstance(op,ast.Pow):
            if not isinstance(b,(int,float)) or abs(b)>16:raise ProgramError('Exponent magnitude must be <=16')
            return self.bounded(a**b)
        raise ProgramError('Unsupported arithmetic operator')

    def assign(self,node,value,local):
        if isinstance(node,ast.Name):local[node.id]=value
        elif isinstance(node,(ast.Tuple,ast.List)):
            if len(node.elts)!=len(value):raise ProgramError('Unpack length mismatch')
            for n,v in zip(node.elts,value):self.assign(n,v,local)
        elif isinstance(node,ast.Subscript):
            obj=self.expression(node.value,local);key=self.expression(node.slice,local)
            if not isinstance(obj,(dict,list)):raise ProgramError('Only dict/list items can be assigned')
            obj[key]=value;self.bounded(obj)
        else:raise ProgramError('Unsupported assignment target')

    def statement(self,node,local):
        self.tick()
        if isinstance(node,ast.Return):raise Returned(self.expression(node.value,local) if node.value else None)
        if isinstance(node,ast.Assign):
            value=self.expression(node.value,local)
            for target in node.targets:self.assign(target,value,local)
        elif isinstance(node,ast.AugAssign):self.assign(node.target,self.binary(node.op,self.expression(node.target,local),self.expression(node.value,local)),local)
        elif isinstance(node,ast.If):
            for stmt in node.body if self.expression(node.test,local) else node.orelse:self.statement(stmt,local)
        elif isinstance(node,(ast.For,ast.While)):
            if node.orelse:raise ProgramError('Loop else is unsupported')
            iterator=iter(self.expression(node.iter,local)) if isinstance(node,ast.For) else None
            while True:
                self.tick()
                if iterator is not None:
                    try:value=next(iterator)
                    except StopIteration:break
                    self.assign(node.target,value,local)
                elif not self.expression(node.test,local):break
                try:
                    for stmt in node.body:self.statement(stmt,local)
                except BreakLoop:break
                except ContinueLoop:continue
        elif isinstance(node,ast.Expr):self.expression(node.value,local)
        elif isinstance(node,ast.Pass):pass
        elif isinstance(node,ast.Break):raise BreakLoop()
        elif isinstance(node,ast.Continue):raise ContinueLoop()
        else:raise ProgramError('Unsupported statement: '+type(node).__name__)

    def expression(self,node,local):
        self.tick()
        e=lambda n:self.expression(n,local)
        if isinstance(node,ast.Constant):
            if type(node.value) not in (int,float,str,bool,type(None)):raise ProgramError('Unsupported constant')
            return self.bounded(node.value)
        if isinstance(node,ast.Name):
            if node.id in local:return local[node.id]
            if node.id in self.globals:return self.globals[node.id]
            raise ProgramError('Unknown variable: '+node.id)
        if isinstance(node,ast.List):return self.bounded([e(v) for v in node.elts])
        if isinstance(node,ast.Tuple):return self.bounded(tuple(e(v) for v in node.elts))
        if isinstance(node,ast.Dict):
            if any(k is None for k in node.keys):raise ProgramError('Use dict.copy/update rather than ** expansion')
            return self.bounded({e(k):e(v) for k,v in zip(node.keys,node.values)})
        if isinstance(node,ast.Subscript):return e(node.value)[e(node.slice)]
        if isinstance(node,ast.Slice):return slice(e(node.lower) if node.lower else None,e(node.upper) if node.upper else None,e(node.step) if node.step else None)
        if isinstance(node,ast.BinOp):return self.binary(node.op,e(node.left),e(node.right))
        if isinstance(node,ast.UnaryOp):
            value=e(node.operand)
            if isinstance(node.op,ast.USub):return -value
            if isinstance(node.op,ast.UAdd):return +value
            if isinstance(node.op,ast.Not):return not value
        if isinstance(node,ast.BoolOp):
            value=e(node.values[0])
            for n in node.values[1:]:
                if isinstance(node.op,ast.And) and not value or isinstance(node.op,ast.Or) and value:return value
                value=e(n)
            return value
        if isinstance(node,ast.IfExp):return e(node.body) if e(node.test) else e(node.orelse)
        if isinstance(node,ast.Compare):
            a=e(node.left)
            for op,n in zip(node.ops,node.comparators):
                b=e(n)
                tests={ast.Eq:lambda:a==b,ast.NotEq:lambda:a!=b,ast.Lt:lambda:a<b,ast.LtE:lambda:a<=b,ast.Gt:lambda:a>b,ast.GtE:lambda:a>=b,ast.In:lambda:a in b,ast.NotIn:lambda:a not in b,ast.Is:lambda:a is b,ast.IsNot:lambda:a is not b}
                if type(op) not in tests or not tests[type(op)]():return False
                a=b
            return True
        if isinstance(node,ast.Attribute):
            if isinstance(node.value,ast.Name) and node.value.id=='math' and node.attr in ('pi','e','tau'):return getattr(math,node.attr)
            raise ProgramError('Attribute access is limited to math constants and approved method calls')
        if isinstance(node,(ast.ListComp,ast.DictComp)):
            result=[] if isinstance(node,ast.ListComp) else {}
            scope=local.copy()
            def walk(index):
                self.tick()
                if index==len(node.generators):
                    if isinstance(result,list):result.append(self.expression(node.elt,scope))
                    else:result[self.expression(node.key,scope)]=self.expression(node.value,scope)
                    self.bounded(result);return
                g=node.generators[index]
                if g.is_async:raise ProgramError('async is unsupported')
                for value in self.expression(g.iter,scope):
                    self.tick();self.assign(g.target,value,scope)
                    if all(self.expression(cond,scope) for cond in g.ifs):walk(index+1)
            walk(0);return result
        if isinstance(node,ast.Call):
            if any(isinstance(a,ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):raise ProgramError('Argument unpacking unsupported')
            args=[e(a) for a in node.args];kwargs={k.arg:e(k.value) for k in node.keywords}
            if isinstance(node.func,ast.Name):
                name=node.func.id
                if name in self.functions:
                    if kwargs:raise ProgramError('Helper functions use positional arguments')
                    return self.call_function(name,args)
                builtins={'abs':abs,'min':min,'max':max,'sum':sum,'len':len,'range':range,'enumerate':lambda x:list(enumerate(x)),'zip':lambda *x:list(zip(*x)),'int':int,'float':float,'bool':bool,'round':round,'list':list,'dict':dict,'tuple':tuple,'sorted':sorted,'all':all,'any':any}
                if name not in builtins:raise ProgramError('Unavailable function: '+name)
                return self.bounded(builtins[name](*args,**kwargs))
            if isinstance(node.func,ast.Attribute):
                name=node.func.attr;base=node.func.value
                if isinstance(base,ast.Name) and base.id=='math':
                    if name not in ('sin','cos','tan','asin','acos','atan','atan2','sqrt','exp','log','floor','ceil','fabs','hypot','degrees','radians','isfinite','copysign'):raise ProgramError('Unsupported math function')
                    return getattr(math,name)(*args,**kwargs)
                obj=e(base)
                allowed={dict:('get','copy','items','keys','values','update','setdefault','pop'),list:('copy','append','pop')}
                if name not in allowed.get(type(obj),()):raise ProgramError('Unavailable method: '+name)
                result=getattr(obj,name)(*args,**kwargs);self.bounded(obj)
                return self.bounded(list(result) if name in ('items','keys','values') else result)
        raise ProgramError('Unsupported expression: '+type(node).__name__)
