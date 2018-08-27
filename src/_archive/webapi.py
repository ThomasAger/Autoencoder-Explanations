
########### Python 3.2 #############
import http.client, urllib.request, urllib.parse, urllib.error, base64, json

headers = {
    # Request headers
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': '1b689dc09b714f469b71b7c0d2b70038',
}


def getTotalWordArray(main, secondary, tertiary):
    words = []
    for m in main:
        words.append(m)
        for s in secondary:
            words.append(m + " " + s)
            for t in tertiary:
                words.append(m + " " + s + " " + t)
    return words

def generateJsonQueries(word, templates):
    json_queries = '{"queries":['
    for t in range(len(templates)):
        d = {
            'words': templates[t],
            'word': word,
        }
        json_queries = json_queries + json.dumps(d)
        if t != len(templates)-1:
            json_queries += ","
    json_queries = json_queries + ']}'
    return json_queries



def getProbabilityScore(word):

    main_words = ["movie", "film"]
    secondary_words = ["as", "is", "was", "has", "it", "a", "like", "that"]
    tertiary_words = ["a", "more", "is", "have"]

    templates = getTotalWordArray(main_words, secondary_words, tertiary_words)

    params = urllib.parse.urlencode({
        # Request parameters
        'model': 'body',
        'order': '5'
    })

    body = '{"queries":[{'

    try:
        conn = http.client.HTTPSConnection('api.projectoxford.ai')
        conn.request("POST", "/text/weblm/v1.0/calculateConditionalProbability?%s" % params, generateJsonQueries(word, templates), headers)
        response = conn.getresponse()
        data = response.read()
        parsed_json = json.loads(str(data)[2:-1])
        highest_proba = -21470000.0
        highest_template = ""
        for key, value in parsed_json.items():
            for v in value:
                prob = v['probability']
                if prob > highest_proba:
                    highest_proba = prob
                    highest_template = v['words'] + " " + v['word']
        conn.close()
        return highest_template, highest_proba
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

def getHighestScore(words):
    highest_prob = -21470000.0
    highest_temp = ""
    highest_index = 0
    for w in range(len(words)):
        temp, prob = getProbabilityScore(words[w])
        if prob > highest_prob:
            highest_prob = prob
            highest_temp = temp
            highest_index = w
    return highest_temp, highest_index

