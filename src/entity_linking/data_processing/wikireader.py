# Wikipedia dump parsing code
# Original version from: https://github.com/matthewfl/nlp-entity-convert
# This code is licensed under LGPL v3 https://www.gnu.org/licenses/lgpl-3.0.html

import re
import json
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
#import bz2
import math

#import regex
from pymongo import MongoClient
import pdb
class WikipediaReader(object):

    title_rg = re.compile('.*<title>(.*)</title>.*')
    id_rg = re.compile('.*<id>(.*)</id>.*')
    link_rg = re.compile('\[\[([^\]]*)\]\]')
    redirect_rg = re.compile('.*<redirect title="(.*)" />')
    not_link_match = re.compile('[^a-zA-Z0-9_]')
    page_namespace_rg = re.compile('.*<ns>(.*)</ns>.*')

    def __init__(self, fname):
        self.wikidump_fname = fname

    def read(self):
        current_page = None
        look_for_next_page = True
        page_text = None
        page_namespace = 0
        page_id = -1
        title_rg = self.title_rg
        look_for_next_id = True

        with open(self.wikidump_fname, encoding='utf8') as f:
            line = '<init>'
            while line:
                line = f.readline()
                if look_for_next_page:
                    if '<page>' not in line:
                        continue
                    else:
                        look_for_next_page = False
                if '<title>' in line:
                    current_page = title_rg.match(line).group(1)
                elif '<redirect' in line:
                    redirect_page = self.redirect_rg.match(line).group(1)
                    self.readRedirect(current_page, redirect_page, page_namespace)
                    look_for_next_page = True
                    look_for_next_id = True
                elif '<id>' in line:
                    if look_for_next_id:
                        page_id = self.id_rg.match(line).group(1)
                        look_for_next_id = False
                elif '<ns>' in line:
                    page_namespace = int(self.page_namespace_rg.match(line).group(1))
                elif '<text' in line:
                    lines = [ line[line.index('>')+2:] ]
                    if '</text>' in lines[0]:
                        page_text = lines[0][:lines[0].index('</text>')]
                        look_for_next_page = True
                        look_for_next_id = True
                        self.readPage(current_page, page_id, page_text, page_namespace)
                    else:
                        while line:
                            line = f.readline()
                            if not line:
                                break
                            if '</text>' in line:
                                lines.append(line[:line.index('</text>')])
                                look_for_next_page = True
                                look_for_next_id = True
                                page_text = '\n'.join(lines)
                                self.readPage(current_page, page_id, page_text, page_namespace)
                                break
                            else:
                                lines.append(line)


    @classmethod
    def getLinkTargets(cls, content):
        ret = cls.link_rg.findall(content)
        def s(v):
            a = v.split('|')
            pg = a[0].replace(' ', '_').replace('(', '_lrb_').replace(')', '_rrb_').lower()
            pg = cls.not_link_match.sub('', pg)
            txt = a[-1]
            if '://' not in v:
                return pg, txt
        return [a for a in [s(r) for r in ret] if a is not None]

    def readPage(self, title, page_id, content, namespace):
        pass

    def readRedirect(self, title, target, namespace):
        pass


class WikiRegexes(object):

    redirects = {}

    page_titles = set()

    _wiki_re_pre = [
        (re.compile('&amp;'), '&'),
        (re.compile('&lt;'), '<'),
        (re.compile('&gt;'), '>'),
        (re.compile('<ref.+?<\/ref>'), ''),
        (re.compile('<.*?>'), ''),
        (re.compile('\[http[^\] ]*', re.IGNORECASE), ''),
        (re.compile('[a-zA-Z]+:\/\/[^\]\} ]+', re.IGNORECASE), ''),
        (re.compile('\|(thumb|left|right|\d+px)', re.IGNORECASE), ''),
        (re.compile('\[\[image:[^\[\]]*\|([^\[\]]*)\]\]', re.IGNORECASE), '\\l'),
        (re.compile('\[\[category:([^\|\]\[]*)[^\]\[]*\]\]', re.IGNORECASE), '[[\\1]]'),  # make category into links
        (re.compile('\[\[[a-z\-]*:[^\]]\]\]'), ''),
        #(re.compile('\[\[[^\|\]]*\|'), '[['),
        #(regex.compile('\{((?R)|[^\{\}]*)*\}'), ''),  # this is a recursive regex
        (re.compile('{{[^\{\}]*}}'), ''),
        (re.compile('{{[^\{\}]*}}'), ''),
        (re.compile('{{[^\{\}]*}}'), ''),
        (re.compile('{[^\{\}]*}'), ''),
        (re.compile('{[^\{\}]*}'), ''),
        (re.compile('{[^\{\}]*}'), ''),
    ]

    _wiki_re_post = [
        (re.compile('[\[|\]]'), ''),
        (re.compile('&[^;]*;'), ' '),
        (re.compile('\(|\)'), ''),
        #(re.compile('\)'), '_rrb_'),
        (re.compile('\n+'), ' '),
        # (re.compile(' \d+ '), ' ### '),  # numbers on their own in text are replaces with ###, maintain numbers in page titles
        # (re.compile('^\d+ '), '### '),
    ]

    _wiki_re_text = [(re.compile('[^a-zA-Z0-9_ .,?!&%$()\-/\'\":;@]'), ''), (re.compile('\s+'), ' ')]

    _wiki_re_text_punc = [
        (re.compile('[^a-zA-Z0-9_ ]'), ''),
        (re.compile(' \d+ '), ' ### '),
        (re.compile('^\d+ '), '### '),
        (re.compile('\s+'), ' ')
    ]

    _wiki_links_to_text = [
        (re.compile('\[\[([^\|\]\[\{\}]+?)\|([^\]\[\{\}]*)\]\]'), '\\2'),
        (re.compile('\[\[([^\[\|\n\]]*)\]\]'), '\\1'),
    ]

    _wiki_re_all = _wiki_re_pre + _wiki_links_to_text + _wiki_re_post + _wiki_re_text
    _wiki_re_all_punc = _wiki_re_pre + _wiki_links_to_text + _wiki_re_post + _wiki_re_text_punc

    _wiki_link_re = [
        re.compile('\[\[([^\|\n\]]*)\]\]'),
        re.compile('\[\[([^\|\]\[\{\}]+?)\|([^\]\[\{\}]*)\]\]'),
    ]

    _wiki_non_title = re.compile('[^a-z0-9_]')

    def _wikiResolveLink(self, match):
        # print(match.groups())
        #import ipdb; ipdb.set_trace()
        m = match.group(1)
        if m:
            mg = self.convertToTitle(m)
            tit = self.redirects.get(mg, mg)
            if tit in self.page_titles:
                return tit
            else:
                return match.group(0)
        else:
            return match.group(0)

    @classmethod
    def convertToTitle(cls, tit):
        return cls._wiki_non_title.sub('', tit.replace(' ', '_').replace('(', '_lrb_').replace(')', '_rrb_').lower())

    @classmethod
    def _wikiToText(cls, txt, strip_punc_num=False):
        # sents = sent_tokenize(txt.lower())
        # text = []
        # for sent in sents:
        #     for r in cls._wiki_re_all:
        #         sent = r[0].sub(r[1], sent)
        #     text.append(sent)
        # return text
        txt = txt.lower()
        if strip_punc_num:
            for r in cls._wiki_re_all_punc:
                txt = r[0].sub(r[1], txt)
        else:
            for r in cls._wiki_re_all:
                txt = r[0].sub(r[1], txt)
        return txt

    def _wikiToLinks(self, txt):
        sents = sent_tokenize(txt)
        text = []
        for sent in sents:
            txt = sent.lower()
            print(txt)
            for r in self._wiki_re_pre:
                txt = r[0].sub(r[1], txt)
            print(txt)
            for r in self._wiki_link_re:
                txt = r.sub(self._wikiResolveLink, txt)
            print(txt)
            for r in self._wiki_re_post:
                txt = r[0].sub(r[1], txt)
            print(txt)
            text.append(txt)
        print(text)
        return text

    def _wikiToInstances(self, page_id, txt, max_context=40):
        lines = txt.split('\n')
        instances = []
        half_context = math.floor(max_context/2)
        for line in lines:
            txt = line
            line_instances = []
            for r in self._wiki_re_pre:
                txt = r[0].sub(r[1], txt)
            for r in self._wiki_links_to_text:
                match = r[0].search(txt)
                while match is not None:
                    num_groups = len(match.groups())
                    left = txt[:match.start()]
                    right = txt[match.end():]
                    if num_groups == 2:
                        line_instances.append([page_id, match.groups()[0], match.groups()[1], left, right])
                    else:
                        line_instances.append([page_id, match.groups()[0], match.groups()[0], left, right])
                    txt = r[0].sub(r[1], txt, 1)
                    match = r[0].search(txt)

            for r in self._wiki_re_post:
                txt = r[0].sub(r[1], txt)
            for inst in line_instances:
                left = inst[3]
                right = inst[4]
                for r in self._wiki_links_to_text:
                    right = r[0].sub(r[1], right, 1)
                for r in self._wiki_re_post:
                    left = r[0].sub(r[1], left)
                    right = r[0].sub(r[1], right)
                left = ' '.join(word_tokenize(left)[-half_context:])
                right = ' '.join(word_tokenize(right)[:half_context])
                # left = word_tokenize(inst[0])[:-20]
                # right = word_tokenize(inst[1])[:20]
                inst[3] = left
                inst[4] = right
                instances.append(inst)
        return instances


class WikipediaW2VParser(WikipediaReader, WikiRegexes):

    def __init__(self, wiki_fname, redirect_fname, surface_count_fname, output_fname, mongo_db=None, json_dir=None):
        super(WikipediaW2VParser, self).__init__(wiki_fname)
        self.redirect_fname = redirect_fname
        self.output_fname = output_fname
        self.surface_count_fname = surface_count_fname
        self.read_pages = False
        self.build_train = False
        self.redirects = json.load(open('../resources/wikipedia/wiki_redirects.json'))
        self.page_titles = set()
        self.surface_to_title = defaultdict(lambda: defaultdict(lambda: 0))
        self.num_read = 0
        self.db = None
        self.json_dir = json_dir
        self.json_out = None
        self.num_instances = 0
        self.inst_per_file = 500000
        self.num_files = 0
        if mongo_db:
            self.client = MongoClient("localhost", 27017)
            self.db = self.client[mongo_db]
            self.docs = self.db.docs
            self.docs.create_index('t')
            self.docs.create_index('ct')

    def _wikiResolveLink(self, match):
        page = super(WikipediaReader, self)._wikiResolveLink(match)
        surface = match.groups()[-1]
        if '[' not in surface and '[' not in page:
            self.surface_to_title[surface][page] += 1
        return page

    def save_redirects(self):
        cnt = 0
        cont_iters = True
        print("Beginning redirect tracing")
        # resolve double or more redirects
        while cnt < 10 and cont_iters:
            cont_iters = False
            cnt += 1
            print("Beginning iteration {}")
            count = 0
            for k, v in self.redirects.items():
                count += 1
                if count % 100 == 0:
                    print("Processed {} redirect items".format(count))
                v2 = self.redirects.get(v)
                if v2:
                    cont_iters = True
                    self.redirects[k] = v2

        with open(self.redirect_fname, 'w+') as f:
            json.dump(self.redirects, f)

    def save_surface_counts(self):
        with open(self.surface_count_fname, 'w+') as f:
            json.dump(self.surface_to_title, f)

    def readRedirect(self, title, target, namespace):
        if not self.read_pages:
            # self.redirects[self.convertToTitle(title)] = self.convertToTitle(target)
            self.redirects[title] = target

    def readPage(self, title, page_id, content, namespace):
        if namespace != 0:
            return  # ignore pages that are not in the core of wikipedia
        self.num_read += 1
        if self.num_read % 10000 == 0:
            print("Read {} articles".format(self.num_read))
        if self.read_pages:
            if self.db:
                # exit(1)
                self.docs.update_one({"_id": page_id}, {"$set": {"untok_text": self._wikiToText(content)}})
            else:
                content = self._wikiToText(content)
                self.save_f.write(content)
                self.save_f.write('\n')
        elif self.build_train:
            instances = self._wikiToInstances(page_id, content, max_context=128)
            for inst in instances:
                json.dump(inst, self.json_out)
                self.json_out.write('\n')
                self.num_instances += 1
                if self.num_instances > self.inst_per_file:
                    self.json_out.close()
                    self.num_files += 1
                    self.num_instances = 0
                    print("Moving to instance file {}".format(self.num_files))
                    self.json_out = open(self.json_dir + '/instances_{}.json'.format(self.num_files), 'w+')

        else:
            self.page_titles.add(self.convertToTitle(title))

    def run(self):
        # read the reidrects first
        # print("Reading redirects")
        # self.read_pages = False
        # self.read()
        # self.save_redirects()
        # self.read_pages = True
        self.build_train = True
        self.json_out = open(self.json_dir + '/instances_{}.json'.format(self.num_files), 'w+')
        self.num_read = 0
        print("Reading text and mention counts")
        # self.save_f = open(self.output_fname, 'w+')
        self.read()
        # self.save_f.close()
        # self.json_out.close()
        # self.save_surface_counts()



def main():
    # wikipedia_raw_dump output_redirects output_surface_counts output_text
    import sys
    parser = WikipediaW2VParser(sys.argv[-1], None, None, None, #sys.argv[-3], sys.argv[-2], sys.argv[-1],
                                # json_dir='../resources/json-instances',
                                json_dir="../resources/wikiEL/json-instances/full-untok")
    parser.run()


if __name__ == '__main__':
    main()