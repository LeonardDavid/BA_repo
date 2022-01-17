class Test
{
    public:
        inline Test(const int number, const char character) : d_number(number), d_character(character)
        {
            // equivalent to " : d_number(number), d_character(character) "
            // d_number = number
            // d_character = character;
        }
        
        int getNumber(){
            return d_number;
        }

        char getCharacter(){
            return d_character;
        }

        inline ~Test(){}

    protected:
        int d_number;
        char d_character;
};