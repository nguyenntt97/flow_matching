% sice cls file for LuaLaTeX and uplatex
% © [2023/08/31] [Tadaaki Sone]
% Minor adjustments by [2025/09/09] [Akira Furui]

% ref:https://www.sice-si.org/conf/si2025/paper.php

\ProvidesClass{sice-si}[2025/09/09 sice-si]

\newif\if@sice@biblatex
\@sice@biblatexfalse
\DeclareOption{biblatex}{\@sice@biblatextrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{jlreq}}
\ProcessOptions\relax

% (u)pLaTeX非互換パッケージに対して互換させるパッチを読み込む命令
% LuaLaTeXでは不要
\RequirePackage{plautopatch}

\NeedsTeXFormat{LaTeX2e}

% 2カラム,A4,フォントサイズ10pt
\LoadClass[
    twocolumn,
    paper=a4,
    fontsize=10pt,
    disablejfam,
    head_space=20mm,
    foot_space=27mm,
    gutter=15mm,
    fore-edge=15mm,
    column_gap=7.5mm,
    linegap=.5zh,]
    {jlreq}

\RequirePackage{ifptex}
\RequirePackage{ifluatex}


\PassOptionsToPackage{hyphens}{url}
\RequirePackage{secdot} % sectionの数字の後のドットの制御
\RequirePackage{setspace} % 行間設定用パッケージ
\setstretch{0.8} % 行間設定
\sectiondot{section}
% \sectionpunct{section}{.} % replaces \sectiondot{section} % sectionの数字の後のドットの有効化
\RequirePackage{caption} % キャプションとサブキャプションのパッケージ
\RequirePackage{subcaption}
\RequirePackage{here} % 好きな位置にfigなどを置くためのパッケージ
\RequirePackage[hyphens]{xurl} % URLを表示するためのパッケージ、ハイフンでの改行を許可
\urlstyle{same} % URLの表示形式を本文と同じにする
\RequirePackage{enumitem} % 高機能番号付き箇条書きのパッケージ
\RequirePackage{amssymb, amsmath} % 高度な数式記述用パッケージ
\RequirePackage{mathtools} % 高度な数式記述用パッケージ
\RequirePackage{newtxtext,newtxmath} % Times系フォントを使用するためのパッケージ
\RequirePackage{siunitx} % SI単位記法に準拠した形式で単位を書くためのパッケージ
\RequirePackage{indentfirst} % 最初の段落でインデントするためのパッケージ

\ifluatex
    % LuaTeXを日本語で使うための設定
    \RequirePackage{luatexja}
    % 日本語フォントは原ノ味フォントを使用
    \RequirePackage{luatexja-fontspec}
    \RequirePackage[haranoaji,no-math]{luatexja-preset}

    % 図表関連のパッケージを読み込む
    \RequirePackage{graphicx}

    % ハイパーリンクを設定
    \RequirePackage[unicode,hidelinks,pdfusetitle]{hyperref}
    \RequirePackage{bookmark}
\else
    \ifuptex
        % upLaTeXのフォントを設定
        \RequirePackage[haranoaji,noalphabet]{pxchfon}

        % 図表関連のパッケージを読み込む
        \RequirePackage{color}
        \RequirePackage[dvipdfmx]{graphicx}

        % ハイパーリンクを設定
        \RequirePackage[dvipdfmx,unicode,hidelinks,pdfusetitle]{hyperref}

    \else
        % pTeXでコンパイルしようとする場合はエラーを出す
        \ClassError{sice}{This class file is for LuaLaTeX or upLaTeX only.}{Use LuaLaTeX or upLaTeX to compile this document.}
    \fi
\fi

\if@sice@biblatex
    \RequirePackage[style=sice,sorting=none,date=year,isbn=false,doi,giveninits,urldate=iso,seconds]{biblatex} % biblatexを使用するためのパッケージ
    \defbibheading{bibliography}{\section*{参考文献}}
\else
    \RequirePackage{xpatch}
    % \PassOptionsToPackage{super}{cite}
    % \AtBeginDocument{%
    %     \@ifpackageloaded{cite}{%
    %         \renewcommand\citeform[1]{#1)}%
    %     }{}%
    %     \xpatchcmd{\thebibliography}{\labelsep=1\jlreq@zw}{\labelsep=.5\jlreq@zw}{}{}%
    % }
    \AtBeginDocument{%
        \xpatchcmd{\thebibliography}{\labelsep=1\jlreq@zw}{\labelsep=.5\jlreq@zw\itemsep=0pt\parsep=0pt}{}{}%
    }
\fi


%ページ数の書式
\pagestyle{empty}

% figとtableの余白の調整
% \setlength{\intextsep}{0pt}

% キャプションとサブキャプションの設定
\RequirePackage{silence}
\WarningFilter{caption}{Unknown document class (or package)}
\RequirePackage{caption}
\RequirePackage{subcaption}
\captionsetup[figure]{labelsep=quad, labelformat=simple, name=\bfseries Fig., skip=5pt, font={stretch=0.8}, belowskip=0pt}
\captionsetup[subfigure]{labelsep=space, labelformat=simple, name=\bfseries Fig., skip=3pt, font={stretch=0.8}, belowskip=0pt}
\renewcommand*{\thesubfigure}{(\alph{subfigure})}
\captionsetup[table]{labelsep=quad, labelformat=simple, name=\bfseries Table, skip=5pt, font={stretch=0.8}, belowskip=0pt}

% 本文中のFigureとTableの表記統一のため，\Fig・\Tableを定義
% 引数としてlabelを受け取り，それを使ってFigureやTableを参照する
% \FigTableマクロの定義
\newcounter{figcount}
\newcommand{\FigTable}[3][]{%
    \def\temp{#1}%
    \ifx\temp\empty
        \ifcsdef{#2:\detokenize{#3}}%
        {#2~\ref{#3}}%
        {\csgdef{#2:\detokenize{#3}}{}%
        \textbf{#2~\ref{#3}}}%
    \else
        \textbf{#2~\ref{#3}}%
    \fi
}

% \Figマクロの定義
\newcommand{\Fig}[2][]{%
    \FigTable[#1]{Fig.}{#2}%
}

% \Figureマクロの定義
\newcommand{\Figure}[2][]{%
    \FigTable[#1]{Figure}{#2}%
}

% \Tableマクロの定義
\newcommand{\Table}[2][]{%
    \FigTable[#1]{Table}{#2}%
}

% sectionのフォントサイズと行送りを調整 (fontsize{size}{baselineskip})
\RenewBlockHeading{section}{1}{font={\fontsize{12pt}{15pt}\selectfont\bfseries\raggedright}}
\RenewBlockHeading{subsection}{2}{font={\fontsize{11pt}{14pt}\selectfont\bfseries\raggedright}}
\RenewBlockHeading{subsubsection}{3}{font={\fontsize{10pt}{13pt}\selectfont\bfseries\raggedright}}
% セクション前後の空白の設定 (after_spaceを2ptから5ptに改善)
% \ModifyHeading{section}{before_space=8pt, after_space=5pt}
% \ModifyHeading{subsection}{before_space=7pt, after_space=4pt}
% \ModifyHeading{subsubsection}{before_space=6pt, after_space=3pt}

\ModifyHeading{section}{before_space=8pt, after_space=5pt, after_label_space=0.1em}
\ModifyHeading{subsection}{before_space=7pt, after_space=4pt}
\ModifyHeading{subsubsection}{before_space=6pt, after_space=3pt, after_label_space=0.1em}

% セクションが連続する際の行取りを設定
\SetBlockHeadingSpaces{
    {_section{*},_subsection{before_space=0pt, after_space=2pt}}
    {_subsection{*},_subsubsection{before_space=0pt, after_space=2pt}}
}

% タイトルの変数設定
\def\title#1{\gdef\@title{#1}}
\let\@title\@empty

% 著者名の変数設定
\def\name#1{\gdef\@name{#1}}
\let\@name\@empty

% 英語タイトルの変数設定
\def\etitle#1{\gdef\@etitle{#1}}
\let\@etitle\@empty

% 英語著者名の変数設定
\def\ename#1{\gdef\@ename{#1}}
\let\@ename\@empty

% アブストラクトの変数設定
\def\abst#1{\gdef\@abst{#1}}
\let\@abst\@empty

% タイトルの設定
\makeatletter
\renewcommand{\@maketitle}{\newpage
    \centering
    \vspace{12pt}

    % もしtitleが定義されているならtitleを表示
    \ifx\@title\@empty\else
        \fontsize{16pt}{32pt}\selectfont
        \@title
        \par
        \vspace{3pt}
    \fi
    
    % もしnameが定義されているならnameを表示
    \ifx\@name\@empty\else
        \fontsize{12pt}{27pt}\selectfont
        \@name
        \par
    \fi

    % もしetitleが定義されているならetitleを表示
    \ifx\@etitle\@empty\else
        \fontsize{16pt}{32pt}\selectfont
        \begin{bfseries}
            \@etitle
        \end{bfseries}
        \par
        \vspace{3pt}
    \fi

    % もしenameが定義されているならenameを表示
    \ifx\@ename\@empty\else
        \fontsize{12pt}{27pt}\selectfont
        \@ename
        \par
    \fi

    \vspace{8pt}

    % abstractの設定
    \begin{minipage}{160mm}
        \fontsize{9pt}{9pt}\selectfont
        Abstract: \@abst
    \end{minipage}
    \vspace{18pt}
}
\makeatother
