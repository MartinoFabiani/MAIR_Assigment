import abc
import random
import pandas as pd
from Levenshtein import distance

# abstract class which provides a template to create different classifiers
class Classifier(abc.ABC):
    @abc.abstractmethod
    def train(self, sentences, labels):
        pass
    
    @abc.abstractmethod
    def predict(self, sentence):
        pass
    
    def batch_predict(self, sentences):
        vectorized_predict_function = np.vectorize(self.predict)
        predictions = vectorized_predict_function(sentences)
        return predictions

# for initial testing purposes, the rule based classifier is used instead of a ML model
class RuleBasedClassifier(Classifier):
    
    def train(self, sentences, labels):
        self.default = 'inform'
        self.rules = [
            (lambda sentence: 'kay ' in sentence, 'ack'),
            (lambda sentence: 'yes' in sentence, 'affirm'),
            (lambda sentence: 'goodbye' in sentence, 'bye'),
            (lambda sentence: 'is it' in sentence, 'confirm'),
            (lambda sentence: 'does it' in sentence, 'confirm'),
            (lambda sentence: 'do they' in sentence, 'confirm'),
            (lambda sentence: 'i dont want' in sentence, 'deny'),
            (lambda sentence: 'hello' in sentence, 'hello'),
            (lambda sentence: 'hi i am' in sentence, 'hello'),
            (lambda sentence: 'looking for' in sentence, 'inform'),
            (lambda sentence: 'no' in sentence, 'negate'),
            (lambda sentence: 'again' in sentence, 'repeat'),
            (lambda sentence: 'go back' in sentence, 'repeat'),
            (lambda sentence: 'anything else' in sentence, 'reqalts'),
            (lambda sentence: 'how about' in sentence, 'reqalts'),
            (lambda sentence: 'more' in sentence, 'reqmore'),
            (lambda sentence: 'phone number' in sentence, 'request'),
            (lambda sentence: 'address' in sentence, 'request'),
            (lambda sentence: 'start over' in sentence, 'restart'),
            (lambda sentence: 'thank you' in sentence, 'thankyou'),
            (lambda sentence: 'noise' in sentence, 'null'),
            (lambda sentence: 'cough' in sentence, 'null'),
            (lambda sentence: 'unintelligible' in sentence, 'null'),
        ]

    def predict(self, sentence):
        
        for rule in self.rules:
            matches_rule, label = rule
            
            if matches_rule(sentence):
                return label
        
        # if none of the rules match
        return self.default


# class which holds all of the keyword matching and preference extraction logic
class KeywordMatcher:

    def __init__(self):
        restaurant_info = pd.read_csv('restaurant_info.csv')
        self.restaurant_info = restaurant_info
        self.price_ranges = set(restaurant_info['pricerange'].dropna())
        self.areas = set(restaurant_info['area'].dropna())
        self.food_types = set(restaurant_info['food'].dropna())
        self.two_word_food_types = [food_type for food_type in self.food_types if ' ' in food_type]
        
        self.corrected_categories = set()

    def register_correction(self, correction):
        if correction in self.price_ranges:
            self.corrected_categories.add('price_range')
        
        if correction in self.areas:
            self.corrected_categories.add('areas')
    
        if correction in self.food_types or correction in self.two_word_food_types:
            self.corrected_categories.add('food_type')

    def levehnstein(self, input, category):
        correction = False
        min_distance = 100
        for entry in category:
            l_distance = distance(input, entry)
            if l_distance <= 3 and l_distance < min_distance:
                correction = entry
        self.register_correction(correction)
        return correction

    def check_categories(self, utterance, specific_category=False):
        self.corrected_categories = set()
        restaurant_info = self.restaurant_info
        price_ranges = self.price_ranges
        areas = self.areas
        food_types = self.food_types
        two_word_food_types = self.two_word_food_types
        levehnstein = self.levehnstein
    
        '''Returns dictionary of categories, False if not recognized'''
        categories = {
            'price_range': False,
            'area': False,
            'food_type': False
        }
        utterance_split = utterance.split()

        if specific_category in categories.keys() and \
            any([keyword in utterance for keyword in ['any', 'dont care', "don't care", 'whatever',
                                                      'doesnt matter', "doesn't matter"]]):
            categories[specific_category] = 'dontcare'


        # Check price range
        for price_range in price_ranges:
            if price_range in utterance:
                categories['price_range'] = price_range
                break
        if 'moderately' in utterance:
            categories['price_range'] = 'moderate'
        if not categories['price_range']:
            if len(utterance_split) == 1:
                categories['price_range'] = levehnstein(utterance, price_ranges)
            elif any([keyword in utterance_split for keyword in ['price', 'priced', 'restaurant']]):
                if 'restaurant' in utterance_split:
                    idx = utterance_split.index('restaurant')
                if 'price' in utterance_split or 'priced' in utterance_split:
                    idx = utterance_split.index('price') if 'price' in utterance_split else utterance_split.index('priced')
                categories['price_range'] = levehnstein(utterance_split[idx-1], price_ranges)

        # Check area
        for area in areas:
            if area in utterance:
                # Exception for north american
                if area == 'north' and 'north' in utterance_split \
                        and distance(utterance_split[utterance_split.index('north')+1], 'american') <= 3:
                    continue
                categories['area'] = area
                break
        if 'center' in utterance:
            categories['area'] = 'centre'
        if not categories['area']:
            if len(utterance_split) == 1:
                categories['area'] = levehnstein(utterance, areas)
            elif 'part' in utterance_split or 'area' in utterance_split:
                idx = utterance_split.index('part') if 'part' in utterance_split else utterance_split.index('area')
                categories['area'] = levehnstein(utterance_split[idx-1], areas)
                if utterance_split[idx-1] == 'any':
                    categories['area'] = 'dontcare'

        # Check food type
        for food_type in food_types:
            if food_type in utterance:
                categories['food_type'] = food_type
                break
        if not categories['food_type']:
            if len(utterance_split) == 1:
                categories['food_type'] = levehnstein(utterance, food_types)
            elif 'food' in utterance_split or 'restaurant' in utterance_split:
                idx = utterance_split.index('food') if 'food' in utterance_split else utterance_split.index('restaurant')
                if idx == 1:
                    categories['food_type'] = levehnstein(utterance_split[idx-1], food_types)
                else:
                    categories['food_type'] = levehnstein(' '.join(utterance_split[idx-2:idx]), two_word_food_types)
                    if not categories['food_type']:
                        categories['food_type'] = levehnstein(utterance_split[idx-1], food_types)
            elif len(utterance_split) == 2:
                categories['food_type'] = levehnstein(utterance, two_word_food_types)
                
        # indicate which catagories where corrected using levehnstein distance
        categories['corrections'] = list(self.corrected_categories)
        return categories


class DialogManager:

    def __init__(self, classifier: Classifier, keywordMatcher: KeywordMatcher):
        self.classifier = classifier
        self.keywordMatcher = keywordMatcher
        self.restaurant_info = pd.read_csv('restaurant_info.csv')
        self.config = {
            'pref_change_allowed': True,
            'debug_mode': True
        }
        self.clear_internal_state()

    # indicates whether the dialog has ended or not
    def dialog_is_ongoing(self):
        return self.current_state != 'terminated'
    
    # register the system response with the self.most_recent_response property, and
    # modify the system response if required
    def respond(self, system_repsonse):
        if self.config['debug_mode']:
            for internal_update in self.latest_internal_updates:
                print('DEBUG: internal update: {}'.format(internal_update))
            print('DEBUG: num possible recommendations: {}'.format(len(self.recommendation_list)))
            print('DEBUG: current state: {}'.format(self.current_state))
    
        self.most_recent_response = system_repsonse
        return system_repsonse
    
    # resets the dialog manager's internal state to its initial state
    def clear_internal_state(self):
        self.current_state = 'welcome'
        self.required_clarification = None
        self.most_recent_response = ''
        self.latest_internal_updates = []
        self.current_recommendation = ''
        self.recommendation_list = []
        self.user_preferences = {
            'price': None,
            'area': None,
            'food': None
        }

    # uses the user's current preferences as recorded in the self.user_preferences
    # to find all the restaurants that match, and stores the names of those restaurants
    # in self.recommendation_list
    def update_recommendation_list(self):
        any_keyword = 'dontcare'
        conditions = []
    
        price = self.user_preferences['price']
        if price != None and price != any_keyword:
            conditions.append(f'pricerange=="{price}"')
        
        area = self.user_preferences['area']
        if area != None and area != any_keyword:
            conditions.append(f'area=="{area}"')
        
        food = self.user_preferences['food']
        if food != None and food != any_keyword:
            conditions.append(f'food=="{food}"')
        
        query = ' and '.join(conditions)
        self.recommendation_list = self.restaurant_info.query(query)['restaurantname'].tolist()
        
    # given an 'inform' user utterance, all preferences are extracted from the utterance
    # using a keyword matching algorithm and these preferences are the stored in
    # self.user_preferences
    def update_user_preferences(self, inform_utterance):
        cat = False
        if self.current_state == 'awaiting-price-pref':
            cat = 'price_range'
        if self.current_state == 'awaiting-area-pref':
            cat = 'area'
        if self.current_state == 'awaiting-food-pref':
            cat = 'food_type'
    
        extracted_preferences = self.keywordMatcher.check_categories(inform_utterance, cat)
        
        if len(extracted_preferences['corrections']) > 0:
            category = extracted_preferences['corrections'][0]
            corrected_value = extracted_preferences[category]
            self.required_clarification = (category, corrected_value)
            return False
        
        if extracted_preferences['price_range'] != False:
            self.user_preferences['price'] = extracted_preferences['price_range']
            self.latest_internal_updates.append('extracted price preference: {}'.format(extracted_preferences['price_range']))
        
        if extracted_preferences['area'] != False:
            self.user_preferences['area'] = extracted_preferences['area']
            self.latest_internal_updates.append('extracted area preference: {}'.format(extracted_preferences['area']))
        
        if extracted_preferences['food_type'] != False:
            self.user_preferences['food'] = extracted_preferences['food_type']
            self.latest_internal_updates.append('extracted food preference: {}'.format(extracted_preferences['food_type']))
    
        return True
    
    # handles the selection of a recommendation and the formatting of the natural
    # language response of the recommendation. if a previous recommendation was made
    # the recommendation is discarded in favor of the new one
    def provide_new_recommendation(self):
        if self.current_recommendation in self.recommendation_list:
            self.recommendation_list.remove(self.current_recommendation)
    
        self.current_recommendation = random.choice(self.recommendation_list)
        return '{} is a great restaurant'.format(self.current_recommendation)
    
    # join the given phrases together in a natural language friendly manner, i.e.,
    # each of the phrases is seperated by a comma, expept the last two, which are
    # separated by the word 'and'
    def join_language_phrases(self, phrases):
        if len(phrases) == 1:
            return phrases[0]
        else:
            return (', '.join(phrases[:-1]) + ' and ' + phrases[-1])

    # handles the retrieving and formating of requested information of the current
    # recommended restaurant given a user utterance clasified as 'request' and
    # returns a system response with all the requested information
    def provide_requested_information(self, request_utterance):
        name = self.recommendation_list[0]
        info = self.restaurant_info.set_index('restaurantname').loc[name]
    
        utterance_words = set(request_utterance.split())
        information_phrases = []
        
        # TODO: handle when the data is not present in the info database
        if len(utterance_words & {'price', 'pricerange'}) > 0:
            information_phrases.append('is in the {} price range'.format(info['pricerange']))
            
        if len(utterance_words & {'area'}) > 0:
            information_phrases.append('is a nice place in the {} of town'.format(info['area']))
            
        if len(utterance_words & {'food'}) > 0:
            information_phrases.append('servers {} food'.format(food = info['food']))
            
        if len(utterance_words & {'phone', 'phonenumber'}) > 0:
            information_phrases.append('can be reached at {}'.format(info['phone']))
            
        if len(utterance_words & {'address'}) > 0:
            information_phrases.append('is located on {}'.format(info['addr']))
            
        if len(utterance_words & {'postcode'}) > 0:
            information_phrases.append('has the following postcode: {}'.format(info['postcode']))
        
        if len(information_phrases) > 0:
            return (name + ' ' + self.join_language_phrases(information_phrases))
        else:
            return 'My apologies, I did not understand. Consider rephrasing.'


    # handles the updating of user preferences based on a user utterance
    # classified as 'inform', this function also determines if enough information
    # is present to make a recommendation or if more information is required
    def handle_inform_utterance(self, inform_utterance):
        if not self.update_user_preferences(inform_utterance):
            category, corrected_value = self.required_clarification
            self.current_state = 'clarification-needed'
            if category == 'price_range':
                return 'You are looking for a restaurant in the {} price range right?'.format(corrected_value)
            if category == 'area':
                return 'You are looking for a restaurant in the {} part of town right?'.format(corrected_value)
            if category == 'food_type':
                return 'You are looking for a restaurant that serves {} food right?'.format(corrected_value)
            return 'Did you mean {}'.format(corrected_value)
        
        self.update_recommendation_list()
                        
        if len(self.recommendation_list) == 0:
            if self.config['pref_change_allowed']:
                self.current_state = 'awaiting-new-pref'
                return 'I am sorry but there is no restaurant matching your criteria. Would you like to look for a different restaurant?'
            else:
                self.current_state = 'terminated'
                return 'I am sorry but there is no restaurant matching your criteria. Please restart.'
    
        if len(self.recommendation_list) <= 3:
            self.current_state = 'made-recommendation'
            return self.provide_new_recommendation()
        
        if self.user_preferences['price'] == None:
            self.current_state = 'awaiting-price-pref'
            return 'Would you like something in the cheap, moderate, or expensive price range?'
        
        if self.user_preferences['area'] == None:
            self.current_state = 'awaiting-area-pref'
            return 'What part of town do you have in mind?'
        
        if self.user_preferences['food'] == None:
            self.current_state = 'awaiting-food-pref'
            return 'What kind of food would you like?'
            
        self.current_state == 'made-recommendation'
        return self.provide_new_recommendation()


    # given an user utterance, the dialog manager will generate a system response based
    # on its internal state, and if required the internal state of the dialog manger
    # will be updated to reflect the content of the user utterance
    def get_system_response(self, user_utterance):
        if user_utterance == None:
            return self.respond('Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?')
    
        dialog_act = self.classifier.predict(user_utterance)
        self.latest_internal_updates = ['speech act predication: {}'.format(dialog_act)]
        
        if dialog_act in ['bye', 'thankyou']:
            self.current_state = 'terminated'
            return self.respond('Thank you, have a great day.')
        
        if dialog_act == 'restart':
            self.current_state = 'welcome'
            self.clear_internal_state()
            return self.respond('Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?')
    
        if dialog_act == 'repeat':
            return self.respond(self.most_recent_response)
        
        if self.current_state == 'clarification-needed':
            if dialog_act == 'inform':
                self.required_clarification = None
                return self.respond(self.handle_inform_utterance(user_utterance))
            
            if dialog_act in ['ack', 'affirm', 'confirm']:
                corrected_value = self.required_clarification[1]
                self.required_clarification = None
                return self.respond(self.handle_inform_utterance(corrected_value))
            
            self.required_clarification = None
            if self.user_preferences['price'] == None:
                self.current_state = 'awaiting-price-pref'
            
            if self.user_preferences['area'] == None:
                self.current_state = 'awaiting-area-pref'
                
            if self.user_preferences['food'] == None:
                self.current_state = 'awaiting-food-pref'
        
        if dialog_act == 'inform':
            if self.current_state != 'made-recommendation' or self.config['pref_change_allowed']:
                return self.respond(self.handle_inform_utterance(user_utterance))
            
            return self.respond('I am sorry, but you cannot change your preferences at this time.')
        
        
        if self.current_state == 'welcome':
            self.current_state = 'awaiting-price-pref'
            return self.respond('Would you like something in the cheap, moderate, or expensive price range?')
        
        if self.current_state == 'awaiting-price-pref':
            return self.respond('Would you like something in the cheap, moderate, or expensive price range?')
        
        if self.current_state == 'awaiting-area-pref':
            return self.respond('What part of town do you have in mind?')
        
        if self.current_state == 'awaiting-food-pref':
            return self.respond('What kind of food would you like?')
        
        if self.current_state == 'awaiting-new-pref':
            if dialog_act in ['ack', 'affirm', 'confirm']:
                return self.respond('Please specify your new preferences.')
            
            self.current_state = 'terminated'
            return self.respond('I am sorry I could not help you today.')
        
        if self.current_state == 'made-recommendation':
            if dialog_act in ['reqalts', 'reqmore', 'deny', 'negate']:
                if len(self.recommendation_list) > 1:
                    return self.respond(self.provide_new_recommendation())
                    
                if self.config['pref_change_allowed']:
                    self.current_state = 'awaiting-new-pref'
                    return self.respond('I am sorry but there is no other restaurants matching your criteria. Would you like to look for a different restaurant?')
                else:
                    self.current_state = 'terminated'
                    return self.respond('I am sorry but there is no restaurant matching your criteria. Please restart.')
            
            if dialog_act == 'request':
                return self.respond(self.provide_requested_information(user_utterance))
    
            return self.respond('Would you like some additional information for {}?'.format(self.current_recommendation))
        
        
        return self.respond('My apologies, I did not understand. Consider rephrasing.')


def main():
    # some parts of the dialog manager utilize random numbers, seed the random number
    # generator to make sure each series of dialogs progress exactly the same
    random.seed(27)
    
    with open('all_dialogs.txt', 'r') as file_handle:
        task = random.choice([line.rstrip() for line in file_handle if line.startswith('Task')])
    print(task)
    
    # initialize the keyword match algorithm and the dialog classifier to be
    # used by the dialog manager
    keywordMatcher = KeywordMatcher()
    classifier = RuleBasedClassifier()
    classifier.train(None, None)
    
    while(True):
        dialog_manager = DialogManager(classifier, keywordMatcher)
        user_utterance = None
        
        system_repsonse = dialog_manager.get_system_response(user_utterance)
        print('System:', system_repsonse)
        
        while(dialog_manager.dialog_is_ongoing()):
            user_utterance = input('User: ').lower()
            system_repsonse = dialog_manager.get_system_response(user_utterance)
            print('System:', system_repsonse, '\n')
        
        if input('Announcement: the dialog has ended. Start a new dialog? [y/n]') != 'y':
            break
 

if __name__ == '__main__':
    main()

