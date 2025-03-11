// Create House enum to better map house names
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum House {
    Gryffindor,
    Hufflepuff,
    Ravenclaw,
    Slytherin,
}

impl House {
    pub fn from_str(s: &str) -> Option<House> {
        match s {
            "Gryffindor" => Some(House::Gryffindor),
            "Hufflepuff" => Some(House::Hufflepuff),
            "Ravenclaw" => Some(House::Ravenclaw),
            "Slytherin" => Some(House::Slytherin),
            _ => None,
        }
    }

    pub fn from_index(index: i32) -> Option<&'static str> {
        match index {
            0 => Some("Gryffindor"),
            1 => Some("Hufflepuff"),
            2 => Some("Ravenclaw"),
            3 => Some("Slytherin"),
            _ => None,
        }
    }

    pub fn to_index(self) -> i32 {
        match self {
            House::Gryffindor => 0,
            House::Hufflepuff => 1,
            House::Ravenclaw => 2,
            House::Slytherin => 3,
        }
    }
}

// Implement Display trait for House enum, doesn't have to be public
use std::fmt;
impl fmt::Display for House {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let house_name = match *self {
            House::Gryffindor => "Gryffindor",
            House::Hufflepuff => "Hufflepuff",
            House::Ravenclaw => "Ravenclaw",
            House::Slytherin => "Slytherin",
        };
        write!(f, "{}", house_name)
    }
}
