"""Predict the class membership of a Wikipedia edit.
The edit is either vandalism or an edit made in good faith. For this assignment
you may ignore the possibility of edge cases.
"""

import requests,re,nltk
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction import text
# from sklearn.naive_bayes import MultinomialNB



WIKIPEDIA_API_ENDPOINT = 'https://en.wikipedia.org/w/api.php'


arrBad = [ '2g1c', '2 girls 1 cup', 'acrotomophilia', 'anal', 'anilingus', 'anus', 'arsehole', 'ass', 'asshole', 'assmunch', 'auto erotic', 'autoerotic', 'babeland', 'baby batter', 'ball gag', 'ball gravy', 'ball kicking', 'ball licking', 'ball sack', 'ball sucking', 'bangbros', 'bareback', 'barely legal', 'barenaked', 'bastardo', 'bastinado', 'bbw', 'bdsm', 'beaver cleaver', 'beaver lips', 'bestiality', 'bi curious', 'big black', 'big breasts', 'big knockers', 'big tits', 'bimbos', 'birdlock', 'bitch', 'black cock', 'blonde action', 'blonde on blonde action', 'blow j', 'blow your l', 'blue waffle', 'blumpkin', 'bollocks', 'bondage', 'boner', 'boob', 'boobs', 'booty call', 'brown showers', 'brunette action', 'bukkake', 'bulldyke', 'bullet vibe', 'bung hole', 'bunghole', 'busty', 'butt', 'buttcheeks', 'butthole', 'camel toe', 'camgirl', 'camslut', 'camwhore', 'carpet muncher', 'carpetmuncher', 'chocolate rosebuds', 'circlejerk', 'cleveland steamer', 'clit', 'clitoris', 'clover clamps', 'clusterfuck', 'cock', 'cocks', 'coprolagnia', 'coprophilia', 'cornhole', 'cum', 'cumming', 'cunnilingus', 'cunt', 'darkie', 'date rape', 'daterape', 'deep throat', 'deepthroat', 'dick', 'dildo', 'dirty pillows', 'dirty sanchez', 'dog style', 'doggie style', 'doggiestyle', 'doggy style', 'doggystyle', 'dolcett', 'domination', 'dominatrix', 'dommes', 'donkey punch', 'double dong', 'double penetration', 'dp action', 'eat my ass', 'ecchi', 'ejaculation', 'erotic', 'erotism', 'escort', 'ethical slut', 'eunuch', 'faggot', 'fecal', 'felch', 'fellatio', 'feltch', 'female squirting', 'femdom', 'figging', 'fingering', 'fisting', 'foot fetish', 'footjob', 'frotting', 'fuck', 'fucking', 'fuck buttons', 'fudge packer', 'fudgepacker', 'futanari', 'g-spot', 'gang bang', 'gay sex', 'genitals', 'giant cock', 'girl on', 'girl on top', 'girls gone wild', 'goatcx', 'goatse', 'gokkun', 'golden shower', 'goo girl', 'goodpoop', 'goregasm', 'grope', 'group sex', 'guro', 'hand job', 'handjob', 'hard core', 'hardcore', 'hentai', 'homoerotic', 'honkey', 'hooker', 'hot chick', 'how to kill', 'how to murder', 'huge fat', 'humping', 'incest', 'intercourse', 'jack off', 'jail bait', 'jailbait', 'jerk off', 'jigaboo', 'jiggaboo', 'jiggerboo', 'jizz', 'juggs', 'kike', 'kinbaku', 'kinkster', 'kinky', 'knobbing', 'leather restraint', 'leather straight jacket', 'lemon party', 'lolita', 'lovemaking', 'make me come', 'male squirting', 'masturbate', 'menage a trois', 'milf', 'missionary position', 'motherfucker', 'mound of venus', 'mr hands', 'muff diver', 'muffdiving', 'nambla', 'nawashi', 'negro', 'neonazi', 'nig nog', 'nigga', 'nigger', 'nimphomania', 'nipple', 'nipples', 'nsfw images', 'nude', 'nudity', 'nympho', 'nymphomania', 'octopussy', 'omorashi', 'one cup two girls', 'one guy one jar', 'orgasm', 'orgy', 'paedophile', 'panties', 'panty', 'pedobear', 'pedophile', 'pegging', 'penis', 'phone sex', 'piece of shit', 'piss pig', 'pissing', 'pisspig', 'playboy', 'pleasure chest', 'pole smoker', 'ponyplay', 'poof', 'poop chute', 'poopchute', 'porn', 'porno', 'pornography', 'prince albert piercing', 'pthc', 'pubes', 'pussy', 'queaf', 'raghead', 'raging boner', 'rape', 'raping', 'rapist', 'rectum', 'reverse cowgirl', 'rimjob', 'rimming', 'rosy palm', 'rosy palm and her 5 sisters', 'rusty trombone', 's&m', 'sadism', 'scat', 'schlong', 'scissoring', 'semen', 'sex', 'sexo', 'sexy', 'shaved beaver', 'shaved pussy', 'shemale', 'shibari', 'shit', 'shota', 'shrimping', 'slanteye', 'slut', 'smut', 'snatch', 'snowballing', 'sodomize', 'sodomy', 'spic', 'spooge', 'spread legs', 'strap on', 'strapon', 'strappado', 'strip club', 'style doggy', 'suck', 'sucks', 'suicide girls', 'sultry women', 'swastika', 'swinger', 'tainted love', 'taste my', 'tea bagging', 'threesome', 'throating', 'tied up', 'tight white', 'tit', 'tits', 'titties', 'titty', 'tongue in a', 'topless', 'tosser', 'towelhead', 'tranny', 'tribadism', 'tub girl', 'tubgirl', 'tushy', 'twat', 'twink', 'twinkie', 'two girls one cup', 'undressing', 'upskirt', 'urethra play', 'urophilia', 'vagina', 'venus mound', 'vibrator', 'violet blue', 'violet wand', 'vorarephilia', 'voyeur', 'vulva', 'wank', 'wet dream', 'wetback', 'white power', 'women rapping', 'wrapping men', 'wrinkled starfish', 'xx', 'xxx', 'yaoi', 'yellow showers', 'yiffy', 'zoophilia']

def page_ids(titles):
    """Look up the Wikipedia page ids by title.

    For example, the Wikipedia page id of "Albert Einstein" is 736. (This page
    has a small page id because it is one of the very first pages on
    Wikipedia.)

    A useful reference for the Mediawiki API is this page:
    https://www.mediawiki.org/wiki/API:Info

    Args:
        titles (list of str): List of Wikipedia page titles.

    Returns:
        list of int: List of Wikipedia page ids.

    """
    # The following lines of code (before `YOUR CODE HERE`) are suggestions
    params = {
        'action': 'query',
        'prop': 'info',
        'titles': '|'.join(titles),
        'format': 'json',
        'formatversion': 2,  # version 2 is easier to work with
    }
    payload = requests.get(WIKIPEDIA_API_ENDPOINT, params=params).json()
    # YOUR CODE HERE
    retval = []
    for title in titles:
        temp = payload['query']['pages']
        for val in temp:
            retval.append(val['pageid'])
    return retval

def recent_revision_ids(id, n):
    """Find the revision ids of recent revisions to a single Wikipedia page.

    The Wikipedia page is identified by its page id and only the `n` most
    recent revision ids are returned.

    https://www.mediawiki.org/wiki/API:Revisions

    Args:
        id (int): Wikipedia page id
        n (int): Number of revision ids to return.

    Returns:
        list of int: List of length `n` of revision ids.

    """
    # YOUR CODE HERE
    params = {
        'action': 'query',
        'prop': 'revisions',
        'pageids': str(id),
        'format': 'json',
        'formatversion': 2,  # version 2 is easier to work with
        'rvprop' : 'ids',
        'rvlimit':str(n),
    }
    payload = requests.get(WIKIPEDIA_API_ENDPOINT, params=params).json()
    # print(payload)
    retval = []
    allPages = payload['query']['pages']
    for page in allPages:
        for revisions in page['revisions']:
            retval.append(revisions['revid'])

    # print(retval)
    return retval

def revisions(revision_ids):
    """Fetch the content of revisions.

    Revisions are identified by their revision ids.

    https://www.mediawiki.org/wiki/API:Revisions

    Args:
        revision_ids (list of int): Wikipedia revision ids

    Returns:
        list of str: List of revision contents.

    """
    # YOUR CODE HERE
    params = {
        'action': 'query',
        'prop': 'revisions',
        'revids': "|".join(map(str,[id for id in revision_ids])),
        'format': 'json',
        'formatversion': 2,  # version 2 is easier to work with
        'rvprop' : 'ids|content',
    }
    payload = requests.get(WIKIPEDIA_API_ENDPOINT, params=params).json()
    # print(payload)
    retval = []
    allPages = payload['query']['pages']
    for pages in allPages:
        for revisions in pages['revisions']:
            retval.append(revisions['content'])


    # print(retval)
    return retval

def getRawText(revision):
    revision = revision.replace("[", " ")
    revision = revision.replace("]", " ")
    revision = revision.replace("{", " ")
    revision = revision.replace("}", " ")
    revision = revision.replace("'''", " ")
    revision = revision.replace("''", " ")
    revision = re.sub('[!@#$%^&*()|;=.,:-]', ' ', revision)
    return revision


# def is_in_english(quote):
#     d = SpellChecker("en_US")
#     d.set_text(quote)
#     errors = [err.word for err in d]
#     return False if ((len(errors)) > 4 or len(quote.split()) < 3) else True


def is_vandalism(revision_id):
    """Classify a revision as vandalism or non-vandalism.

    For example, revision 738515242 to the page `Indiana` adds the text
    "INdiana best stae ever so ye". This is an uncomplicated case of vandalism.

    This change may be viewed at the following url:

    https://en.wikipedia.org/w/index.php?title=Indiana&diff=738515242&oldid=737943373

    Tip: You may want to compare the parent revision to the current revision.

    Args:
        revision_id (int): Wikipedia revision id

    Returns:
        int: `True` if revision is vandalism, otherwise `False`.

    """
    # YOUR CODE HERE

    # Handle revision
    params1 = {
        'action': 'query',
        'prop': 'revisions',
        'revids': str(revision_id),
        'format': 'json',
        'formatversion': 2,  # version 2 is easier to work with
        'rvprop': 'ids|content',
    }
    payload1 = requests.get(WIKIPEDIA_API_ENDPOINT, params=params1).json()
    doc1 = getRawText(payload1['query']['pages'][0]['revisions'][0]['content'])
    doc1 = doc1.lower().strip()
    doc1Words = nltk.word_tokenize(doc1)
    # doc1Words = doc1.lower().strip().split()
    # doc1Words = [word for word in doc1Words if word not in stopwords.words('english')]
    # doc1Words = [stemmer.stem(word)for word in doc1Words]
    # print(doc1Words)


    # PART 2 - Handle parent
    params2 = {
        'action': 'query',
        'prop': 'revisions',
        'revids': str(payload1['query']['pages'][0]['revisions'][0]['parentid']),
        'format': 'json',
        'formatversion': 2,  # version 2 is easier to work with
        'rvprop': 'ids|content',
    }
    payload2 = requests.get(WIKIPEDIA_API_ENDPOINT, params=params2).json()
    doc2 = getRawText(payload2['query']['pages'][0]['revisions'][0]['content'])
    doc2 = doc2.lower().strip()
    doc2Words = nltk.word_tokenize(doc2)
    # doc2Words = doc2.lower().strip().split()
    # doc2Words = [word for word in doc2Words if word not in stopwords.words('english')]
    # doc2Words = [stemmer.stem(word) for word in doc2Words]
    # print(doc2Words)

    # dict = enchant.Dict("en_US")
    englishVocab = set(w.lower() for w in nltk.corpus.words.words())


    # Create document term matrix for the added words
    corpusAdded = set(doc1Words) - set(doc2Words)
    # vec = text.CountVectorizer()
    # dtm = vec.fit_transform(corpusAdded).toarray()
    # vandals = vec.get_feature_names()
    # dtm = pd.DataFrame(dtm,columns=vandals)
    # print(dtm)

    # Check for profanity and language validity

    for words in corpusAdded:
        if words in arrBad:
            return True
        if words not in englishVocab:
            return True
    return False

    # # Create model and test it against the parent
    # dtmParent = vec.fit_transform(doc2Words).toarray()
    # features = vec.get_feature_names()
    # # dtmParent = pd.DataFrame(dtmParent,columns=vandals)
    # # print(dtmParent)
    #
    # # dtmCurr = vec.fit_transform(doc1Words).toarray()
    # # features = vec.get_feature_names()
    # # dtmCurr = pd.DataFrame(dtmCurr, columns=vandals)
    # # print(dtmCurr)
    #
    # clf = MultinomialNB()
    # clf.fit(dtm,vandals)
    # print(clf.predict(dtmParent))


# DO NOT EDIT CODE BELOW THIS LINE

import unittest


class TestAssignment6(unittest.TestCase):

    def test_page_ids1(self):
        titles = ['Albert Einstein']
        ids = [736]
        self.assertEqual(page_ids(titles), ids)

    def test_recent_revisions1(self):
        n = 3
        id = 736
        revisions = recent_revision_ids(id, n)
        self.assertEqual(len(revisions), n)
        for revid in revisions:
            self.assertGreater(revid, 720000000)

    def test_revisions1(self):
        revids = [746929653]  # edit to Albert Einstein
        revisions_ = revisions(revids)
        self.assertEqual(len(revisions_), len(revids))
        self.assertGreater(len(revisions_[0]), 50000)  # page is more than 50000 characters

    def test_is_vandalism1(self):
        revision_id = 738515242  # clear case of vandalism
        self.assertTrue(is_vandalism(revision_id))


if __name__ == '__main__':
    unittest.main()
