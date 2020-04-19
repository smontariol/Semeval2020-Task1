import random

def filterLine(line, lang, targets):
    if lang=='latin':
        line = ''.join([i for i in line if not (i.isdigit() or i=='#')])
    elif lang=='english':
        wrong_pos = False
        correct_pos = False
        for target in targets:
            line_l = line.split()
            if target in line:
                line = line.replace(target, target[:-3])
                correct_pos = True
            if target[:-3] in line_l:
                wrong_pos = True
        if correct_pos:
            return line
        elif wrong_pos:
            return None
    return line




languages = ['english', 'latin']

for lang in languages:

    output_1 = open(lang + '/' + lang + '_clean_1.txt', 'w', encoding='utf8')
    output_2 = open(lang + '/' + lang + '_clean_2.txt', 'w', encoding='utf8')

    if lang == 'english':
        targets = []
        with open(lang + '/targets.txt', 'r', encoding='utf8') as f:
            for line in f:
                target = line.strip()
                if len(target) > 0 :
                    targets.append(target)
    else:
        targets = None

    with open(lang + '/' + lang + '_1.txt', 'r', encoding='utf8') as f:
        for line in f:
            line = filterLine(line, lang, targets)
            if line is not None:
                output_1.write(line)

    with open(lang + '/' + lang + '_2.txt', 'r', encoding='utf8') as f:
        for line in f:
            line = filterLine(line, lang, targets)
            if line is not None:
                output_2.write(line)

    output_1.close()
    output_2.close()

